# Sources and attribution

- OnlineSPEC: https://github.com/ZinYY/OnlineSPEC at revision
  e58f82eb3f3adca3a686211236bf4f6e9e7e3a2b.
  Paper: https://arxiv.org/abs/2603.12617.
  Selected source files in reference/ are preserved for inspection and numerical
  comparison; they are not imported by production workers.
- The core runtime is a pinned copy of this project's osd_tracedraft backend.
  EAGLE-3 / Transformers source headers and the EAGLE project license are
  preserved in onlinespec_trace/core/backend/.
- OnlineSPEC meta weighting and scheduling are implemented in ensemble.py and
  run.py. The recurrence uses the previously verified local EAGLE-3 implementation.
- Vicuna conversation template: FastChat vicuna_v1.1,
  https://github.com/lm-sys/FastChat/blob/main/fastchat/conversation.py.
- Original OSD background: https://github.com/LiuXiaoxuanPKU/OSD.
  This package's outer algorithm is the OnlineSPEC ensemble, not the earlier
  single-learner OSD experiment.

PROVENANCE.json records upstream and vendored file hashes. No model weights,
tokens, credentials, or complete external repositories are included in the
portable source package.
