"""Configuration stays importable without torch (planning and aggregation)."""
from dataclasses import dataclass, asdict
import math

PROFILES = {
    "deepseek": {"name": "DeepSeek-R1-Distill-Llama-8B", "hidden_size": 4096,
                 "layers": 32, "vocab_size": 128256, "max_length": 4096},
    "vicuna13b": {"name": "vicuna-13b-v1.3", "hidden_size": 5120,
                  "layers": 40, "vocab_size": 32000, "max_length": 2048},
}
METHODS = ("baseline", "tracedraft", "osd", "osd_tracedraft")

@dataclass
class Settings:
    model_profile: str
    method: str
    target: str
    draft: str
    questions: str
    output: str
    seed: int = 0
    order_seed: int | None = None
    max_new_tokens: int = 1024
    max_length: int = 0
    prompt_truncation: str = "left"
    dtype: str = "float16"
    target_quantization: str = "none"
    draft_device: str = "cuda"
    cpu_threads: int = 0
    output_norm_fp32: bool = True
    total_token: int = 60
    depth: int = 5
    top_k: int = 10
    osd_block_size: int = 40
    osd_lr: float = 1e-5
    osd_guard: bool = True
    osd_validation_questions: int = 2
    osd_validation_positions: int = 128
    osd_validation_rtol: float = 0.0
    osd_validation_atol: float = 1e-6
    osd_epochs: int = 2
    osd_batch_size: int = 2
    osd_unroll: int = 7
    osd_decay: float = 0.8
    osd_weight_decay: float = 0.01
    osd_clip: float = 1.0
    osd_loss_chunk: int = 32
    train_last_block: bool = False
    trace_mode: str = "reset"
    trace_lr: float = 1e-5
    trace_validation_guard: bool = True
    trace_validation_rtol: float = 0.0
    trace_validation_atol: float = 1e-6
    trace_diagnostics: str = "basic"
    trace_diagnostics_every: int = 1
    trace_source: str = "balanced"
    trace_update_round: int = 5
    trace_every: int = 1
    trace_prompt_positions: int = 64
    trace_response_positions: int = 64
    replay_mib: float = 0
    replay_positions: int = 0
    anchor_weight: float = 0
    warmup_runs: int = 2
    verify_greedy: int = 0
    verify_tokens: int = 32

    def validate(self):
        if self.model_profile not in PROFILES or self.method not in METHODS:
            raise ValueError("Unknown profile/method")
        if not self.max_length:
            self.max_length = PROFILES[self.model_profile]["max_length"]
        if self.prompt_truncation not in ("left","error"): raise ValueError("prompt_truncation")
        if self.max_new_tokens >= self.max_length-2: raise ValueError("Reserve at least two prompt tokens")
        if self.dtype not in ("float16", "bfloat16", "float32"):
            raise ValueError("Unsupported dtype")
        if self.target_quantization not in ("none", "nf4"):
            raise ValueError("target_quantization must be none or nf4")
        if self.draft_device not in ("cuda", "cpu") or self.cpu_threads < 0:
            raise ValueError("Invalid draft_device or cpu_threads")
        if self.trace_mode not in ("reset", "block_persistent"):
            raise ValueError("Trace mode must be reset or block_persistent")
        if self.trace_diagnostics not in ("basic", "full"):
            raise ValueError("trace_diagnostics must be basic or full")
        if self.trace_source not in ("prompt", "response", "balanced"):
            raise ValueError("Unsupported supervision source")
        for k in ("max_new_tokens", "max_length", "total_token", "depth", "top_k", "osd_block_size",
                  "osd_epochs", "osd_batch_size", "osd_unroll", "osd_loss_chunk", "trace_update_round",
                  "trace_every", "trace_prompt_positions", "trace_response_positions", "verify_tokens",
                  "osd_validation_questions", "osd_validation_positions", "trace_diagnostics_every"):
            if getattr(self, k) < 1: raise ValueError(f"{k} must be positive")
        if self.top_k < 2 or self.total_token < 2 or self.total_token > 1 + self.top_k * (self.depth + 1):
            raise ValueError("Invalid tree budget")
        for k in ("osd_lr", "trace_lr", "osd_clip"):
            if not math.isfinite(getattr(self,k)) or getattr(self,k) <= 0: raise ValueError(k)
        for k in ("osd_weight_decay", "anchor_weight", "replay_mib", "osd_validation_rtol",
                  "osd_validation_atol", "trace_validation_rtol", "trace_validation_atol"):
            if not math.isfinite(getattr(self,k)) or getattr(self,k) < 0: raise ValueError(k)
        if not 0 < self.osd_decay <= 1: raise ValueError("osd_decay must be in (0,1]")
        if min(self.replay_positions, self.warmup_runs, self.verify_greedy) < 0: raise ValueError("Negative count")
        if bool(self.replay_mib) != bool(self.replay_positions):
            raise ValueError("Replay needs both a positive byte budget and sample count")
        if self.trace_mode == "reset" and (self.replay_mib or self.anchor_weight):
            raise ValueError("R/A is supported for block_persistent; keep the Reset control unregularized")
        return self

    @property
    def has_osd(self): return self.method in ("osd", "osd_tracedraft")
    @property
    def has_trace(self): return self.method in ("tracedraft", "osd_tracedraft")
    def to_dict(self): return asdict(self)
