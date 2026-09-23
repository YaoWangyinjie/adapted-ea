"""EA4's independent FP32 head learner. No Transformers/model downloads required.

The frozen draft backbone produces pre-norm features. Only norm/head are adapted,
so these features remain valid for replay and inference KV remains reusable.
"""
from dataclasses import dataclass, asdict
from collections import deque
import math
import random

import torch
import torch.nn.functional as F


@dataclass
class AdaptationConfig:
    learning_rate: float = 1e-6
    temperature: float = 1.0
    scope: str = "head_only"
    source: str = "response"
    prompt_weight: float = 0.25
    max_prompt_positions: int = 128
    max_response_positions: int = 128
    update_at: str = "end"
    warmup_steps: int = 5
    update_every: int = 1  # generation calls, not questions
    train_steps: int = 1
    chunk_size: int = 32
    feature_chunk_size: int = 256
    validation_fraction: float = 0.2
    validation_guard: bool = True
    validation_rtol: float = 0.0
    validation_atol: float = 1e-6
    diagnostics: str = "basic"
    diagnostics_every: int = 1
    gradient_clip: float = 0.2
    max_step_drift: float = 0.02
    max_anchor_drift: float = 0.0  # disabled; establish a diagnostic baseline first
    anchor_weight: float = 0.0
    replay_bytes: int = 0
    replay_positions: int = 64
    precision: str = "master_fp32"  # roundtrip is an intentional negative control
    alignment: str = "correct"  # legacy is an intentional off-by-one control
    seed: int = 0

    def __post_init__(self):
        for name in ("learning_rate", "temperature"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("gradient_clip", "max_step_drift", "max_anchor_drift", "anchor_weight", "validation_rtol", "validation_atol"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        for name in ("max_prompt_positions", "max_response_positions", "warmup_steps",
                     "update_every", "train_steps", "chunk_size", "feature_chunk_size", "diagnostics_every"):
            if not isinstance(getattr(self, name), int) or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("replay_bytes", "replay_positions"):
            if not isinstance(getattr(self, name), int) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if not 0 <= self.prompt_weight <= 1 or not 0 <= self.validation_fraction < 1:
            raise ValueError("Invalid source weight or validation fraction")
        if self.validation_guard and self.validation_fraction == 0:
            raise ValueError("validation_guard requires a held-out fraction")
        if self.diagnostics not in ("basic", "full"):
            raise ValueError("diagnostics must be basic or full")
        if self.source == "balanced" and not 0 < self.prompt_weight < 1:
            raise ValueError("Balanced source requires 0 < prompt_weight < 1; otherwise select a single source")
        for name, choices in {
            "scope": {"head_only", "norm_only", "lm_head_only"},
            "source": {"prompt", "response", "balanced"},
            "update_at": {"prefill", "warmup", "end"},
            "precision": {"master_fp32", "roundtrip"},
            "alignment": {"correct", "legacy"},
        }.items():
            if getattr(self, name) not in choices:
                raise ValueError(f"Unsupported {name}: {getattr(self, name)}")


class PositionReservoir:
    """Bounded, uniform per-source sample; never alters the decoding RNG.

    Stored position j is the teacher logit AFTER consuming x_j. Correct EAGLE
    feature index is j-1, because draft consumes (H_(j-1), x_j).
    """
    def __init__(self, config, vocab_mask, prompt_length, call_index):
        self.config = config
        self.mask = vocab_mask
        self.prompt_length = prompt_length
        self.rows = {"prompt": [], "response": []}
        self.seen = {"prompt": 0, "response": 0}
        self.rng = random.Random(config.seed + 104729 * call_index)

    def _plan(self, start_position, count):
        # Make reservoir decisions before computing/copying teacher logits.
        # The RNG draws and final slots match the original streaming sampler.
        last = {}
        for offset in range(count):
            position = start_position + offset
            if position < 1:
                continue
            group = "prompt" if position < self.prompt_length - 1 else "response"
            if self.config.source not in (group, "balanced"):
                continue
            self.seen[group] += 1
            cap = getattr(self.config, f"max_{group}_positions")
            slot = len(self.rows[group])
            if slot >= cap:
                slot = self.rng.randrange(self.seen[group])
                if slot >= cap:
                    continue
            else:
                self.rows[group].append(None)
            last[group, slot] = (offset, group, slot, position)
        return sorted(last.values(), key=lambda x: x[0])

    @torch.no_grad()
    def _store_selected(self, logits, decisions):
        if logits.ndim == 3:
            logits = logits[0]
        if not decisions:
            return
        mask = self.mask.to(logits.device)
        finite = torch.isfinite(logits).all(-1)
        covered = (finite & mask[logits.argmax(-1)]).cpu().tolist()
        targets = logits[:, mask].cpu()
        masses = logits.float().softmax(-1)[:, mask].sum(-1).cpu().tolist()
        finite = finite.cpu().tolist()
        for i, (_, group, slot, position) in enumerate(decisions):
            self.rows[group][slot] = dict(position=position, group=group, valid=covered[i],
                target=targets[i].clone(), draft_vocab_mass=masses[i] if finite[i] else None)

    @torch.no_grad()
    def observe(self, logits, start_position):
        if logits.ndim == 3:
            logits = logits[0]
        decisions = self._plan(start_position, len(logits))
        if decisions:
            index = torch.tensor([x[0] for x in decisions], device=logits.device)
            self._store_selected(logits.index_select(0, index), decisions)

    @torch.no_grad()
    def observe_prefill(self, hidden, head):
        # Only positions retained by the sampler need an extra target-head call.
        decisions = self._plan(0, hidden.shape[1])
        for start in range(0, len(decisions), self.config.chunk_size):
            batch = decisions[start:start + self.config.chunk_size]
            index = torch.tensor([x[0] for x in batch], device=hidden.device)
            self._store_selected(head(hidden.index_select(1, index)), batch)

    def selected(self, sequence_length):
        offset = 1 if self.config.alignment == "correct" else 0
        return sorted([dict(row, feature_index=row["position"] - offset)
                       for rows in self.rows.values() for row in rows
                       if 0 <= row["position"] - offset < sequence_length - 1],
                      key=lambda row: row["position"])


def adam_candidate(parameters, gradients, state, step, lr, betas=(0.9, 0.999), eps=1e-8):
    """Adam without mutating the live parameters or moments (no weight decay).

    This makes an entire candidate update rejectable without copying the old
    optimizer state. Tested against torch.optim.Adam on multiple steps.
    """
    candidate, moments = {}, {}
    b1, b2 = betas
    with torch.no_grad():
        for name, parameter in parameters.items():
            gradient = gradients[name]
            if name in state:
                old_m, old_v = state[name]
            else:
                old_m, old_v = torch.zeros_like(parameter), torch.zeros_like(parameter)
            m = b1 * old_m + (1 - b1) * gradient
            v = b2 * old_v + (1 - b2) * gradient.square()
            candidate[name] = parameter - lr * (m / (1 - b1 ** step)) / ((v / (1 - b2 ** step)).sqrt() + eps)
            moments[name] = (m, v)
    return candidate, moments


class OnlineHead:
    def __init__(self, norm, head, config):
        self.norm, self.head, self.config = norm, head, config
        self.eps = norm.variance_epsilon
        self.live = {"norm": norm.weight, "weight": head.weight}
        if head.bias is not None:
            self.live["bias"] = head.bias
        names = set(self.live)
        if config.scope == "norm_only":
            names = {"norm"}
        elif config.scope == "lm_head_only":
            names.discard("norm")
        self.names = sorted(names)
        self.initial = {k: v.detach().float().cpu().clone() for k, v in self.live.items()}
        self.master = {k: v.detach().float().clone().requires_grad_(k in names) for k, v in self.live.items()}
        self.moments = {}
        self.step = 0
        self.version = 0
        self.rejected = 0
        self.attempts = 0
        self.history = deque()
        self.history_bytes = 0
        self.rng = random.Random(config.seed + 991)

    def logits(self, features, parameters=None):
        p = self.master if parameters is None else parameters
        x = features.to(device=p["weight"].device, dtype=torch.float32)
        x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + self.eps)
        return F.linear(x * p["norm"], p["weight"], p.get("bias"))

    def reset(self):
        restored = {k: v.to(self.live[k].device).clone() for k, v in self.initial.items()}
        self._publish(restored)
        self.master = {k: v.requires_grad_(k in self.names) for k, v in restored.items()}
        self.moments = {}
        self.step = self.version = self.rejected = 0
        self.history.clear()
        self.history_bytes = 0
        self.rng.seed(self.config.seed + 991)

    @torch.no_grad()
    def _publish(self, parameters):
        # Allocate/cast every new value before copying anything into inference.
        copies = {k: parameters[k].to(self.live[k].dtype) for k in self.names}
        if any(not bool(torch.isfinite(v).all()) for v in copies.values()):
            raise FloatingPointError("Candidate is not representable in inference dtype")
        try:
            for name, value in copies.items():
                self.live[name].copy_(value)
        except Exception:
            for name in self.names:
                self.live[name].copy_(self.master[name].detach().to(self.live[name].dtype))
            raise

    def _source_weights(self, rows):
        counts = {group: sum(r["group"] == group for r in rows) for group in ("prompt", "response")}
        weights = {"prompt": self.config.prompt_weight, "response": 1 - self.config.prompt_weight}
        if self.config.source != "balanced":
            weights = {k: float(k == self.config.source) for k in weights}
        total = sum(weights[k] for k in weights if counts[k])
        if total == 0:
            raise ValueError("No positive-weight training source")
        return {k: weights[k] / total / counts[k] if counts[k] else 0.0 for k in counts}

    def _chunks(self, rows):
        for start in range(0, len(rows), self.config.chunk_size):
            yield rows[start:start + self.config.chunk_size]

    def _loss(self, rows, backward=False, parameters=None, inference=False):
        """Source-normalized KL; chunk graphs are released after each backward."""
        p = self.master if parameters is None else parameters
        weights = self._source_weights(rows)
        total, matches = 0.0, 0
        for chunk in self._chunks(rows):
            features = torch.stack([r["feature"] for r in chunk])
            targets = torch.stack([r["target"] for r in chunk]).to(p["weight"].device).float()
            if inference:
                x = features.to(device=self.head.weight.device, dtype=self.head.weight.dtype)
                logits = self.head(self.norm(x)).float()
            else:
                logits = self.logits(features, p)
            t = self.config.temperature
            per_row = F.kl_div(F.log_softmax(logits / t, -1), F.softmax(targets / t, -1), reduction="none").sum(-1) * t * t
            if not bool(torch.isfinite(per_row).all()):
                raise FloatingPointError("Nonfinite distillation loss")
            scale = per_row.new_tensor([weights[r["group"]] for r in chunk])
            loss = (per_row * scale).sum()
            if backward:
                loss.backward()
            total += float(loss.detach())
            matches += int((logits.argmax(-1) == targets.argmax(-1)).sum())
        return {"kl": total, "top1_agreement": matches / len(rows), "positions": len(rows)}

    @torch.no_grad()
    def evaluate(self, rows, parameters=None, inference=False):
        return self._loss(rows, parameters=parameters, inference=inference) if rows else None

    def _add_history(self, rows):
        for row in rows:
            size = sum(row[k].numel() * row[k].element_size() for k in ("feature", "target"))
            if size > self.config.replay_bytes:
                continue
            self.history.append((row, size))
            self.history_bytes += size
            while self.history_bytes > self.config.replay_bytes:
                _, old_size = self.history.popleft()
                self.history_bytes -= old_size

    def update(self, rows):
        old_rng = self.rng.getstate()
        self.attempts += 1
        full = self.config.diagnostics == "full" and (self.attempts - 1) % self.config.diagnostics_every == 0
        valid = [r for r in rows if r["valid"] and bool(torch.isfinite(r["feature"]).all())
                 and bool(torch.isfinite(r["target"]).all())]
        result = dict(status="skipped", selected_positions=len(rows), valid_positions=len(valid),
                      version_before=self.version, config=asdict(self.config),
                      diagnostics_recorded=full, attempt=self.attempts)
        masses = [r["draft_vocab_mass"] for r in rows if r.get("draft_vocab_mass") is not None]
        result["mean_draft_vocab_mass"] = sum(masses) / len(masses) if masses else None
        train, validation = [], []
        for group in ("prompt", "response"):
            subset = [r for r in valid if r["group"] == group]
            self.rng.shuffle(subset)
            n_val = min(len(subset) - 1, max(1, int(len(subset) * self.config.validation_fraction))) if len(subset) >= 5 and self.config.validation_fraction else 0
            validation.extend(subset[:n_val])
            train.extend(subset[n_val:])
        if not train or (self.config.validation_guard and not validation):
            self.rng.setstate(old_rng)
            result["reason"] = "no_valid_training_positions" if not train else "no_validation_positions"
            return result
        replay = self.rng.sample(list(self.history), min(len(self.history), self.config.replay_positions))
        used = train + [r for r, _ in replay]
        result.update(train_positions=len(train), validation_positions=len(validation), replay_positions=len(replay),
                      valid_by_source={g: sum(r["group"] == g for r in valid) for g in ("prompt", "response")},
                      train_before=None, train_after=None, validation_before=None, validation_after=None,
                      inference_validation_before=None, inference_validation_after=None)
        old_master, old_moments, old_step, old_version = self.master, self.moments, self.step, self.version
        old_history, old_history_bytes = deque(self.history), self.history_bytes
        update_reports = []
        try:
            if self.config.validation_guard or full:
                result["inference_validation_before"] = self.evaluate(validation, inference=True)
            if full:
                result["train_before"] = self.evaluate(train)
                result["validation_before"] = self.evaluate(validation)
            for _ in range(self.config.train_steps):
                for name in self.names:
                    self.master[name].grad = None
                fit = self._loss(used, backward=True)
                anchor_loss = 0.0
                if self.config.anchor_weight:
                    for name in self.names:
                        ref = self.initial[name].to(self.master[name].device)
                        penalty = self.config.anchor_weight * (self.master[name] - ref).square().sum() / ref.square().sum().clamp_min(1e-12)
                        anchor_loss += float(penalty.detach())
                        penalty.backward()
                gradients = {name: self.master[name].grad for name in self.names}
                if any(g is None or not bool(torch.isfinite(g).all()) for g in gradients.values()):
                    raise FloatingPointError("Missing/nonfinite gradient")
                grad_norm = torch.stack([g.norm() for g in gradients.values()]).norm()
                if not bool(torch.isfinite(grad_norm)):
                    raise FloatingPointError("Nonfinite gradient norm")
                if self.config.gradient_clip:
                    scale = (self.config.gradient_clip / (grad_norm + 1e-6)).clamp(max=1)
                    gradients = {k: g * scale for k, g in gradients.items()}
                active = {k: self.master[k] for k in self.names}
                candidate, moments = adam_candidate(active, gradients, self.moments, self.step + 1, self.config.learning_rate)
                report = dict(grad_norm=float(grad_norm), distill_loss=fit["kl"],
                              anchor_loss=anchor_loss, parameters={})
                for name, value in candidate.items():
                    before = self.master[name].detach()
                    cast = value.to(self.live[name].dtype)
                    delta_norm = float((value - before).norm())
                    drift = delta_norm / max(float(before.norm()), 1e-6)
                    anchor_drift = None
                    if self.config.max_anchor_drift:
                        ref = self.initial[name].to(value.device)
                        anchor_drift = float((value - ref).norm() / ref.norm().clamp_min(1e-6))
                    detailed = full or value.numel() <= 4096
                    parameter_report = dict(step_relative_drift=drift, anchor_relative_drift=anchor_drift,
                        master_delta_norm=delta_norm, inference_delta_norm=None,
                        inference_changed_fraction=None, cast_error_norm=None,
                        inference_dtype=str(self.live[name].dtype))
                    if detailed:
                        cast_float = cast.float()
                        parameter_report.update(
                            inference_delta_norm=float((cast_float - self.live[name].float()).norm()),
                            inference_changed_fraction=float((cast != self.live[name]).float().mean()),
                            cast_error_norm=float((value - cast_float).norm()))
                    else:
                        stride = max(1, (value.numel() + 4095) // 4096)
                        sample = cast.flatten()[::stride]
                        parameter_report.update(inference_changed_fraction_sampled=float(
                            (sample != self.live[name].flatten()[::stride]).float().mean()),
                            inference_sample_positions=sample.numel())
                    report["parameters"][name] = parameter_report
                    if not bool(torch.isfinite(value).all()) or not bool(torch.isfinite(cast).all()) or any(not bool(torch.isfinite(x).all()) for x in moments[name]):
                        raise FloatingPointError("Nonfinite/unrepresentable candidate")
                    if self.config.max_step_drift and drift > self.config.max_step_drift:
                        raise FloatingPointError("step_drift_guard")
                    if anchor_drift is not None and anchor_drift > self.config.max_anchor_drift:
                        raise FloatingPointError("anchor_drift_guard")
                merged = {**self.master, **candidate}
                if self.config.precision == "roundtrip":
                    merged = {k: (v.to(self.live[k].dtype).float() if k in self.names else v) for k, v in merged.items()}
                self.master = {k: v.detach().requires_grad_(k in self.names) for k, v in merged.items()}
                self.moments, self.step = moments, self.step + 1
                update_reports.append(report)
                for p in active.values():
                    p.grad = None
                del gradients, active
            # The guard evaluates the actual serving norm/head, including casts.
            # No decoding runs between publication and this transactional check.
            self._publish(self.master)
            if self.config.validation_guard or full:
                result["inference_validation_after"] = self.evaluate(validation, inference=True)
            if self.config.validation_guard:
                before = result["inference_validation_before"]["kl"]
                after = result["inference_validation_after"]["kl"]
                limit = before + self.config.validation_atol + self.config.validation_rtol * abs(before)
                result["validation_limit"] = limit
                if after > limit:
                    raise FloatingPointError("validation_regression")
            if full:
                result["train_after"] = self.evaluate(train)
                result["validation_after"] = self.evaluate(validation)
            self.version += 1
            self._add_history(train)
            result.update(status="updated", version_after=self.version, updates=update_reports)
            return result
        except Exception as exc:
            self.master, self.moments, self.step, self.version = old_master, old_moments, old_step, old_version
            self.history, self.history_bytes = old_history, old_history_bytes
            self.rng.setstate(old_rng)
            self._publish(old_master)
            self.rejected += 1
            if not isinstance(exc, FloatingPointError):
                raise
            result.update(status="rolled_back", reason=str(exc), version_after=self.version,
                          attempted_updates=update_reports, updates=[])
            return result
        finally:
            for value in self.master.values():
                value.grad = None

    def state(self):
        return {"weight_version": self.version, "optimizer_steps": self.step, "rejected_updates": self.rejected, "update_attempts": self.attempts,
                "replay_positions": len(self.history), "replay_bytes": self.history_bytes,
                "trainable_parameters": sum(self.master[n].numel() for n in self.names)}
