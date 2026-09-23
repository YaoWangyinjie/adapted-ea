"""Explicit experiment settings; no dependency on an installed upstream repo."""
from dataclasses import dataclass, asdict
import math
from .core.config import Settings as CoreSettings

@dataclass
class Settings:
    model_profile: str
    target: str
    draft: str
    questions: str
    output: str
    method: str = "onlinespec"
    variant: str = "hedge"
    seed: int = 0
    order_seed: int | None = None
    block_size: int = 40
    lr_1: float = 2e-4
    lr_2: float = 1e-4
    lr_3: float = 4e-4
    epochs: int = 2
    batch_size: int = 2
    weight_decay: float = 0.0
    gradient_clip: float = 1.0
    unroll: int = 7
    decay: float = 0.8
    loss_chunk: int = 32
    hedge_temperature: float = 0.2
    ens_epsilon: float = 0.2
    merge_precision: str = "fp32"
    train_last_block: bool = False
    dtype: str = "float16"
    output_norm_fp32: bool = True
    target_quantization: str = "none"
    draft_device: str = "cuda"
    cpu_threads: int = 0
    max_new_tokens: int = 1024
    max_length: int = 0
    prompt_truncation: str = "left"
    total_token: int = 60
    depth: int = 5
    top_k: int = 10
    warmup_runs: int = 2
    verify_greedy: int = 0
    verify_tokens: int = 32
    trace_mode: str = "reset"
    trace_lr: float = 1e-5
    trace_update_round: int = 5
    trace_every: int = 1
    trace_source: str = "balanced"
    trace_prompt_positions: int = 64
    trace_response_positions: int = 64
    trace_validation_guard: bool = True
    trace_validation_rtol: float = 0.0
    trace_validation_atol: float = 1e-6
    trace_diagnostics: str = "basic"
    trace_diagnostics_every: int = 1
    replay_mib: float = 0.0
    replay_positions: int = 0
    anchor_weight: float = 0.0

    def validate(self):
        if self.method not in ("onlinespec", "onlinespec_trace", "tracedraft", "eagle3"):
            raise ValueError("method must be onlinespec, onlinespec_trace, tracedraft or eagle3")
        if self.variant not in ("hedge", "ens"):
            raise ValueError("variant must be hedge or ens")
        if self.merge_precision not in ("fp32", "upstream_bf16"):
            raise ValueError("merge_precision must be fp32 or upstream_bf16")
        for name in ("lr_1","lr_2","lr_3","hedge_temperature"):
            if not math.isfinite(getattr(self,name)) or getattr(self,name)<=0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.ens_epsilon) or self.ens_epsilon<0:
            raise ValueError("ens_epsilon must be finite and nonnegative")
        inner=self.inner()
        self.max_length=inner.max_length
        return self

    @property
    def learning_rates(self):
        return [self.lr_1,self.lr_2,self.lr_3]

    @property
    def has_outer(self):
        return self.method in ("onlinespec", "onlinespec_trace")

    def inner(self, learner=None):
        fields=CoreSettings.__dataclass_fields__
        values={k:v for k,v in asdict(self).items() if k in fields}
        values.update(
            method="osd_tracedraft" if self.method=="onlinespec_trace" else
                   "tracedraft" if self.method=="tracedraft" else
                   "baseline" if self.method=="eagle3" else "osd",
            osd_block_size=self.block_size,
            osd_lr=self.learning_rates[learner] if learner is not None else self.lr_1,
            osd_epochs=self.epochs,osd_batch_size=self.batch_size,
            osd_weight_decay=self.weight_decay,osd_clip=self.gradient_clip,
            osd_unroll=self.unroll,osd_decay=self.decay,osd_loss_chunk=self.loss_chunk,
            # Preserve OnlineSPEC's outer training protocol in both controls.
            # The optional TraceDraft guard affects only the fast norm/head update.
            osd_guard=False)
        return CoreSettings(**values).validate()

    def to_dict(self):
        return asdict(self)
