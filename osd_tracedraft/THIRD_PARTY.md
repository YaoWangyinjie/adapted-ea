# Source and algorithm provenance

- Original OSD: https://github.com/LiuXiaoxuanPKU/OSD
  inspected revision 788a403d5495896b4fc5b7f56cfd41de5ae61967.
- OnlineSPEC: https://arxiv.org/abs/2603.12617
  local repository inspected revision e58f82eb3f3adca3a686211236bf4f6e9e7e3a2b.
  Relevant files: EAGLE/pipeline_eagle3.py, EAGLE/traineagle3.py,
  EAGLE/train_eagle3/cnets.py, EAGLE/script/EAGLE-3/eagle3-online.sh.
- EAGLE-3/local TraceDraft: ../eagle/model/cnets.py, ea_model_4.py,
  online_head.py, modeling_llama_kv.py. Their source headers and licenses are retained.
- Vicuna template: FastChat's vicuna_v1.1 conversation template.
  https://github.com/lm-sys/FastChat/blob/main/fastchat/conversation.py
- Model specifications:
  https://huggingface.co/lmsys/vicuna-13b-v1.3
  https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B

The recurrent training attention is an independent implementation of the
OnlineSPEC/EAGLE-3 recurrence, with explicit token alignment and masked
current-turn supervision. It is not a claim of byte-for-byte upstream
reproduction. Original OSD's independent causal-LM distillation interface is
not used as an EAGLE draft.

backend/ contains a pinned, unmodified snapshot of nine local EAGLE/TraceDraft
source files and the project LICENSE. SHA256 provenance is recorded in
backend/SNAPSHOT.json. These retain their existing source headers. The package
also uses PyTorch/Transformers. No complete upstream repository or model weights
are bundled.
