"""EA4: observable, persistent head adaptation for the local Llama EAGLE-3 path.

No EA2/EA3 modifications. Initially greedy-only and norm/head-only by design:
these restrictions make target/draft KV reuse and a reference decoder testable.
"""
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
import time

import torch
from torch import nn

from .online_head import AdaptationConfig, OnlineHead, PositionReservoir
from .kv_cache import KVCache


def select_eagle_hidden(states, device, expected_width):
    # Local modeling_llama_kv returns the three selected pre-layer states even
    # when output_hidden_states=False; True additionally appends final norm.
    if states is None or len(states) not in (3, 4):
        raise ValueError("EA4 requires the local Llama KV model's three selected hidden states")
    parts = [x.to(device) for x in states[:3]]
    if any(x.shape[:-1] != parts[0].shape[:-1] for x in parts):
        raise ValueError("Hidden components do not align")
    hidden = torch.cat(parts, dim=-1)
    if hidden.shape[-1] != expected_width:
        raise ValueError(f"Hidden width {hidden.shape[-1]} != {expected_width}")
    return hidden


class EaModel(nn.Module):
    def __init__(self, base_model, draft_model, tokenizer):
        super().__init__()
        self.base_model, self.ea_layer, self.tokenizer = base_model, draft_model, tokenizer
        if base_model.config.model_type != "llama":
            raise ValueError("EA4 currently validates only the local Llama EAGLE-3 path")
        indices = sorted({2, len(base_model.model.layers) // 2, len(base_model.model.layers) - 3})
        if len(indices) != 3:
            raise ValueError("Target needs three distinct EAGLE feature layers")
        self.hidden_layer_indices = indices
        self.use_eagle3 = True
        self.config = base_model.config
        self.base_model.requires_grad_(False)
        self.ea_layer.requires_grad_(False)
        self.eval()
        self.ea_layer.init_tree()
        # cnets.topK_genrate does its d2t lookup on CPU.
        self.ea_layer.d2t = self.ea_layer.d2t.cpu()
        self.ea_layer.t2d = self.ea_layer.t2d.cpu()
        self.vocab_mask = self.ea_layer.t2d.bool()
        if draft_model.lm_head.out_features == base_model.lm_head.out_features:
            self.vocab_mask = torch.ones(base_model.lm_head.out_features, dtype=torch.bool)
        if int(self.vocab_mask.sum()) != draft_model.lm_head.out_features:
            raise ValueError("Checkpoint t2d does not match the draft head")
        mapped = torch.arange(draft_model.lm_head.out_features) + self.ea_layer.d2t
        if not torch.equal(mapped, self.vocab_mask.nonzero().flatten()):
            raise ValueError("Checkpoint d2t ordering disagrees with t2d")
        self.learner = None
        self.adaptation_config = None
        self.calls = 0
        self._cache = None
        self._cache_capacity_value = 0
        self._timings = defaultdict(float)
        self._counts = defaultdict(int)
        self._profile = False

    @classmethod
    def from_pretrained(cls, base_model_path, ea_model_path, total_token=60, depth=5,
                        top_k=10, torch_dtype=torch.float16, device_map="auto", **kwargs):
        # Lazy imports allow CPU-only tests of the adaptation engine.
        from transformers import AutoConfig, AutoTokenizer
        from huggingface_hub import hf_hub_download
        from .modeling_llama_kv import LlamaForCausalLM
        from .configs import EConfig
        from .cnets import Model

        if AutoConfig.from_pretrained(base_model_path).model_type != "llama":
            raise ValueError("Use a Llama target with its matching official EAGLE-3 checkpoint")
        if total_token < 2 or depth < 1 or top_k < 2:
            raise ValueError("Invalid draft tree dimensions")
        target = LlamaForCausalLM.from_pretrained(base_model_path, torch_dtype=torch_dtype,
                                                 device_map=device_map, **kwargs)
        def resolve(name):
            p = Path(ea_model_path) / name
            if p.is_file():
                return str(p)
            if Path(ea_model_path).is_dir():
                raise FileNotFoundError(p)
            return hf_hub_download(ea_model_path, name)
        cfg = EConfig.from_pretrained(resolve("config.json"))
        draft = Model(cfg, load_emb=False, total_tokens=total_token, depth=depth, top_k=top_k)
        # The embedding belongs to the target, and is frozen in this experiment.
        draft.embed_tokens.weight.data.copy_(target.model.embed_tokens.weight.detach().cpu())
        try:
            from safetensors.torch import load_file
            state = load_file(resolve("model.safetensors"), device="cpu")
        except (OSError, EnvironmentError):
            state = torch.load(resolve("pytorch_model.bin"), map_location="cpu", weights_only=True)
        loaded = draft.load_state_dict(state, strict=False)
        missing = set(loaded.missing_keys) - {"embed_tokens.weight"}
        if missing or loaded.unexpected_keys:
            raise ValueError(f"Incompatible draft checkpoint: missing={missing}, unexpected={loaded.unexpected_keys}")
        device = target.model.layers[-1].self_attn.q_proj.weight.device
        draft.to(device=device, dtype=torch_dtype)
        tokenizer = AutoTokenizer.from_pretrained(base_model_path, use_fast=False)
        return cls(target, draft, tokenizer)

    def get_tokenizer(self):
        return self.tokenizer

    @property
    def input_device(self):
        return self.base_model.model.embed_tokens.weight.device

    @property
    def draft_device(self):
        return self.ea_layer.lm_head.weight.device

    def setup_online_adaptation(self, config=None, **kwargs):
        if config is not None and kwargs:
            raise ValueError("Pass either a config or keyword configuration")
        self.adaptation_config = config or AdaptationConfig(**kwargs)
        self.learner = OnlineHead(self.ea_layer.norm, self.ea_layer.lm_head, self.adaptation_config)
        self.calls = 0

    def reset_online_adaptation(self):
        if self.learner is not None:
            self.learner.reset()
        # Scheduling is a stream property and is NOT reset at question boundaries.
        self.ea_layer.reset_kv()

    def get_adaptation_state(self):
        return self.learner.state() if self.learner else {"weight_version": 0, "optimizer_steps": 0}

    def _sync(self):
        devices = {p.device for p in self.parameters() if p.device.type == "cuda"}
        for device in devices:
            torch.cuda.synchronize(device)

    @contextmanager
    def _stage(self, name):
        self._counts[name] += 1
        if self._profile:
            self._sync()
        start = time.perf_counter()
        try:
            yield
        finally:
            if self._profile:
                self._sync()
                self._timings[name] += time.perf_counter() - start

    def _allocate_cache(self, capacity):
        if self._cache is not None and capacity <= self._cache_capacity_value:
            self._cache[2].zero_()
            return self._cache
        # Map actual devices, not CUDA index differences (also works on CPU).
        devices = [layer.self_attn.q_proj.weight.device for layer in self.base_model.model.layers]
        grouped = defaultdict(list)
        for i, device in enumerate(devices):
            grouped[device].append(i)
        lengths = torch.zeros(len(devices) * 2, dtype=torch.long)
        caches, data = [None] * len(devices), []
        cfg = self.config
        dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
        for device, layers in grouped.items():
            buffer = torch.zeros(len(layers) * 2, 1, cfg.num_key_value_heads, capacity, dim,
                                 dtype=self.base_model.dtype, device=device)
            data.append(buffer)
            for local, layer in enumerate(layers):
                caches[layer] = [KVCache(buffer[2 * local + j], lengths[2 * layer + j]) for j in range(2)]
        self._cache = caches, data, lengths
        self._cache_capacity_value = capacity
        return self._cache

    @torch.no_grad()
    def _target(self, ids, cache=None, position_ids=None):
        if position_ids is not None:
            position_ids = position_ids.to(self.input_device)
        out = self.base_model.model(input_ids=ids, past_key_values=cache, position_ids=position_ids,
                                    use_cache=cache is not None, output_hidden_states=False, return_dict=True)
        hidden = select_eagle_hidden(out.hidden_states, self.draft_device, self.ea_layer.fc.in_features)
        return out.last_hidden_state, hidden

    @torch.no_grad()
    def _head(self, hidden):
        return self.base_model.lm_head(hidden.to(self.base_model.lm_head.weight.device))

    def _stops(self, is_llama3):
        ids = getattr(self.tokenizer, "eos_token_id", None)
        stops = set(ids if isinstance(ids, (list, tuple)) else ([] if ids is None else [ids]))
        if is_llama3:
            vocab = self.tokenizer.get_vocab()
            if "<|eot_id|>" in vocab:
                stops.add(vocab["<|eot_id|>"])
        return stops

    @torch.no_grad()
    def _features(self, ids, hidden, collector):
        selected = collector.selected(ids.shape[1])
        if not selected:
            return []
        old_mask = getattr(self.ea_layer, "tree_mask", None)
        try:
            self.ea_layer.tree_mask = None
            # Chunk the FULL contiguous prefix with a separate temporary KV cache.
            # This retains context while bounding the attention matrix; it never
            # overwrites cnets.stable_kv, which belongs to the inference trajectory.
            cache, rows = None, []
            chunk_size = collector.config.feature_chunk_size
            last_index = max(row["feature_index"] for row in selected)
            for begin in range(0, last_index + 1, chunk_size):
                end = min(begin + chunk_size, last_index + 1)
                features, cache = self.ea_layer(hidden[:, begin:end],
                    input_ids=ids[:, begin + 1:end + 1].to(self.draft_device),
                    past_key_values=cache, use_cache=True)
                rows.extend(dict(row, feature=features[0, row["feature_index"] - begin].detach().cpu().clone())
                            for row in selected if begin <= row["feature_index"] < end)
            return rows
        finally:
            self.ea_layer.tree_mask = old_mask

    def _adapt(self, ids, hidden_parts, collector):
        with self._stage("feature_forward"):
            hidden = torch.cat(hidden_parts, dim=1)
            if hidden.shape[1] != ids.shape[1]:
                raise AssertionError("Continuous token/hidden trajectory lost alignment")
            rows = self._features(ids, hidden, collector)
        with self._stage("head_update"):
            result = self.learner.update(rows)
        result["seen_positions"] = dict(collector.seen)
        result["context_tokens"] = ids.shape[1]
        return result

    @torch.no_grad()
    def _draft(self, ids, hidden, next_token):
        return self.ea_layer.topK_genrate(hidden, torch.cat((ids, next_token.to(ids.device)), dim=1),
                                         self.base_model.lm_head, None)

    def eagenerate(self, input_ids, temperature=0.0, max_new_tokens=1024, max_length=4096,
                   log=False, is_llama3=True, enable_adaptation=True, profile=False):
        if temperature != 0:
            raise ValueError("EA4's first validated protocol is greedy; sampling requires a separate correctness study")
        if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] < 2:
            raise ValueError("EA4 expects one prompt containing at least two tokens")
        if max_new_tokens < 0 or max_length <= 0:
            raise ValueError("Invalid generation limits")
        from .utils import evaluate_posterior
        ids = input_ids.to(self.input_device).clone()
        prompt_length = ids.shape[1]
        if prompt_length >= max_length:
            raise ValueError("Prompt leaves no generation capacity")
        self._profile = profile
        self._timings, self._counts = defaultdict(float), defaultdict(int)
        self._sync()
        cuda_devices = {p.device for p in self.parameters() if p.device.type == "cuda"}
        for device in cuda_devices:
            torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        self.calls += 1
        cfg = self.adaptation_config
        do_update = bool(enable_adaptation and self.learner is not None and (self.calls - 1) % cfg.update_every == 0)
        collector = PositionReservoir(cfg, self.vocab_mask, prompt_length, self.calls) if do_update else None
        hidden_parts = []
        adaptation = {"status": "disabled" if not enable_adaptation or self.learner is None else "scheduled_skip"}
        steps = accepted = path_widths = tree_nodes = committed_children = 0
        survival, trials = defaultdict(int), defaultdict(int)
        stop_reason = "max_new_tokens" if max_new_tokens == 0 else None
        stops = self._stops(is_llama3)
        self.ea_layer.reset_kv()
        self.ea_layer.tree_mask = None
        self.base_model.model.tree_mask = None
        self.base_model.model.tree_mode = None
        try:
            if max_new_tokens > 0:
                with self._stage("cache_prepare"):
                    # Extra capacity is verification scratch, not extra returned context.
                    cache, cache_data, lengths = self._allocate_cache(max_length + self.ea_layer.total_tokens + 1)
                with self._stage("target_prefill"):
                    final, hidden = self._target(ids, cache)
                    last_logits = self._head(final[:, -1:])
                if collector is not None:
                    hidden_parts.append(hidden.detach())
                    with self._stage("teacher_collection"):
                        # Dense full-vocabulary logits are transient, chunked, and never replayed.
                        if cfg.source == "response":
                            collector.observe(last_logits, prompt_length - 1)
                        else:
                            collector.observe_prefill(final, self._head)
                    if cfg.update_at == "prefill":
                        adaptation = self._adapt(ids, hidden_parts, collector)
                        collector = None
                        hidden_parts.clear()
                next_token = last_logits[:, -1].argmax(-1, keepdim=True)
                with self._stage("draft_tree"):
                    tree, retrieve, mask, positions = self._draft(ids, hidden, next_token)
                while ids.shape[1] - prompt_length < max_new_tokens and ids.shape[1] < max_length:
                    previous = ids.shape[1]
                    self.base_model.model.tree_mask = mask.to(self.input_device)
                    with self._stage("target_verify"):
                        final, verified_hidden = self._target(tree.to(ids.device), cache, (positions + previous).unsqueeze(0))
                        raw_logits = self._head(final)
                        logits = raw_logits[0, retrieve.to(raw_logits.device)]
                        padded = torch.cat((tree, tree.new_full((1, 1), -1)), dim=1)
                        candidates = padded[0, retrieve.to(tree.device)]
                        best, length, next_distribution = evaluate_posterior(logits, candidates.to(logits.device), None)
                        best, length = int(best), int(length)
                    steps += 1
                    accepted += length
                    path_widths += candidates.shape[1]
                    tree_nodes += tree.numel()
                    for depth in range(1, candidates.shape[1]):
                        trials[depth] += 1
                        survival[depth] += int(length >= depth)
                    count = min(length + 1, max_new_tokens - (previous - prompt_length), max_length - previous)
                    path = candidates[best, :count].to(ids.device)
                    for i, token in enumerate(path.tolist()):
                        if token in stops:
                            path = path[:i + 1]
                            stop_reason = "eos"
                            break
                    count = path.numel()
                    indices = retrieve[best, :count]
                    with self._stage("commit_path"):
                        for buffer in cache_data:
                            src = indices.to(buffer.device) + previous
                            buffer[..., previous:previous + count, :].copy_(buffer.index_select(-2, src))
                        lengths.fill_(previous + count)
                        ids = torch.cat((ids, path.unsqueeze(0)), dim=1)
                        selected_hidden = verified_hidden[:, indices.to(verified_hidden.device)]
                        committed_children += max(0, count - 1)
                    if collector is not None:
                        hidden_parts.append(selected_hidden.detach())
                        with self._stage("teacher_collection"):
                            # No supervision for a hypothetical continuation after EOS.
                            supervised = count - int(stop_reason == "eos")
                            collector.observe(raw_logits[:, indices[:supervised].to(raw_logits.device)], previous)
                        if cfg.update_at == "warmup" and steps >= cfg.warmup_steps:
                            adaptation = self._adapt(ids, hidden_parts, collector)
                            collector = None
                            hidden_parts.clear()
                    if stop_reason == "eos" or ids.shape[1] - prompt_length >= max_new_tokens or ids.shape[1] >= max_length:
                        break
                    next_token = next_distribution.argmax().reshape(1, 1)
                    with self._stage("draft_tree"):
                        # Frozen fc/midlayer means KV is still valid after head/norm updates.
                        # Build the next tree AFTER updating; no stale tree is reused.
                        tree, retrieve, mask, positions = self._draft(ids, selected_hidden, next_token)
                if collector is not None:
                    if cfg.update_at == "end":
                        adaptation = self._adapt(ids, hidden_parts, collector)
                    elif cfg.update_at == "warmup":
                        adaptation = {"status": "skipped", "reason": "warmup_not_reached"}
        finally:
            self.base_model.model.tree_mask = None
            self.base_model.model.tree_mode = None
            self.ea_layer.tree_mask = None
            # Each new generation owns a fresh prefix; no cross-question KV reuse.
            self.ea_layer.reset_kv()
        self._sync()
        elapsed = time.perf_counter() - started
        generated = ids.shape[1] - prompt_length
        stop_reason = stop_reason or ("max_new_tokens" if generated >= max_new_tokens else "context_limit")
        stats = {"total_accept_length": accepted, "total_drafted_tokens": path_widths,
                 "total_steps": steps, "verified_tree_nodes": tree_nodes,
                 "committed_children": committed_children, "new_tokens": generated,
                 "generation_seconds": elapsed, "stop_reason": stop_reason,
                 "depth_trials": dict(trials), "depth_survival": dict(survival),
                 "adaptation": adaptation, "adaptation_state": self.get_adaptation_state(),
                 "profile": profile, "stage_seconds": dict(self._timings) if profile else None,
                 "cuda_peak_allocated_bytes": {str(d): torch.cuda.max_memory_allocated(d) for d in cuda_devices},
                 "stage_calls": dict(self._counts), "hidden_layer_indices": self.hidden_layer_indices}
        return (ids, generated, steps - 1, stats) if log else ids

    @torch.no_grad()
    def naivegenerate(self, input_ids, max_new_tokens=1024, max_length=4096, is_llama3=True):
        """Slow, cache-free greedy reference for short correctness probes only."""
        ids = input_ids.to(self.input_device).clone()
        stops = self._stops(is_llama3)
        self.base_model.model.tree_mask = None
        for _ in range(min(max_new_tokens, max_length - ids.shape[1])):
            out = self.base_model.model(input_ids=ids, use_cache=False, output_hidden_states=False, return_dict=True)
            token = self._head(out.last_hidden_state[:, -1:])[:, -1].argmax(-1, keepdim=True).to(ids.device)
            ids = torch.cat((ids, token), dim=1)
            if int(token.item()) in stops:
                break
        return ids
