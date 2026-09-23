# OnlineSPEC + TraceDraft：三组比较实验

本目录在同一评测流程中比较：

| 结果目录 / `method` | 实验方法 | 外层学习器更新 | 回答内更新 |
|---|---|---|---|
| `onlinespec` | ENS-EAGLE-3 | 开启 | 关闭 |
| `tracedraft` | EAGLE-3 + TraceDraft | 关闭 | 开启 |
| `onlinespec_trace` | ENS-EAGLE-3 + TraceDraft | 开启 | 开启 |

三组共享 target、初始 EAGLE-3 draft、数据和解码设置。独立 TraceDraft 组不会训练 OnlineSPEC 外层学习器。Agent 从用户取得模型路径、数据路径、结果目录和 GPU 编号后，运行下面一个脚本即可，结果自动记录和汇总。

## 1. 安装一次

```bash
git clone https://github.com/YaoWangyinjie/adapted-ea.git
cd adapted-ea
python3.12 -m venv .venv-experiments
source .venv-experiments/bin/activate
python -m pip install -r onlineSpec-trace/requirements.txt
```

已有仓库或已按 [OSD README](../osd_tracedraft/README.md) 安装环境时，直接使用现有仓库和同一个环境。默认未量化 target、全 GPU、每个实验单卡运行。代码后端已包含在本目录，不依赖外部 OnlineSPEC / OSD 克隆，也不需要安装整个 adapted-ea 包。

## 2. 一条命令运行三组

从 `adapted-ea` 仓库根目录执行，填写服务器上的**绝对路径**：

```bash
export PROFILE=deepseek
export TARGET=/absolute/path/DeepSeek-R1-Distill-Llama-8B
export DRAFT=/absolute/path/matching-EAGLE3-DeepSeek-8B
export QUESTIONS=/absolute/path/mt_bench/question.jsonl
export OUT=/absolute/path/results/onlinespec/deepseek_mtbench_seed0

CUDA_VISIBLE_DEVICES=0 bash onlineSpec-trace/scripts/run_comparison.sh --variant hedge
```

脚本按表中顺序运行三组，然后生成同一张比较表。无需分别写三条运行命令，也不会额外跑冻结 EAGLE-3。也可使用 `GPU=0`；同时设置时以 `GPU` 为准。

Vicuna 使用同一个脚本，替换对应路径：

```bash
export PROFILE=vicuna13b
export TARGET=/absolute/path/vicuna-13b-v1.3
export DRAFT=/absolute/path/matching-EAGLE3-vicuna-13b-v1.3
export OUT=/absolute/path/results/onlinespec/vicuna_mtbench_seed0

CUDA_VISIBLE_DEVICES=1 bash onlineSpec-trace/scripts/run_comparison.sh --variant hedge
```

`deepseek` 专用于 DeepSeek-R1-Distill-Llama-8B；`vicuna13b` 专用于 vicuna-13b-v1.3。draft 必须是匹配 target 的 **EAGLE-3** checkpoint，不能使用 EAGLE / EAGLE-2 draft。DeepSeek 默认上下文为 4096，Vicuna 为 2048，默认最多生成 1024 token。

## 3. ENS-EAGLE-3 对应哪条上游管线

这里“ENS-EAGLE-3”指 OnlineSPEC 的三个学习器加权合并方案。上游提供两种不同实现，实验中需固定其一：

| `--variant` | 各块训练起点 | 合并权重 |
|---|---|---|
| `hedge`（默认；上游 README 推荐入口） | 三个学习器都从本块合并模型开始 | 最近一块训练 loss 的 softmax；temperature 0.2 |
| `ens` | 三个学习器分别延续自己的权重和 Adam 状态 | 累计训练 loss 的指数权重；epsilon 0.2 |

若要对齐上游 `pipeline_eagle3_ens.py`，把核心命令改为 `--variant ens` 并使用新的 `OUT`。**不能把 `hedge` 与 `ens` 的基线混在同一比较中。** 本地实现默认以 FP32 混合浮点参数，保留离散词表映射；这是一套共同评测框架下的适配，不是对官方论文数值的直接复现。

外层默认每 40 个完整 conversation 一块；三个学习率分别为 `2e-4`、`1e-4`、`4e-4`，各训练 2 epochs、batch size 2。每个学习器内部使用七步递归损失；**三个学习器不等于七个模型**。训练进程逐一执行，GPU 不同时常驻三个学习器。

TraceDraft 默认 Reset、lr `1e-5`、balanced 监督，prompt / verified response 各最多采样 64 个位置。当前回答的第 5 次验证后尝试一次更新；这不是训练五轮。每个新 conversation 恢复当前块的服务 checkpoint，同一 conversation 的后续 turn 继承临时更新。

例如修改外层和内层学习率：

```bash
CUDA_VISIBLE_DEVICES=0 bash onlineSpec-trace/scripts/run_comparison.sh \
  --variant hedge --lr-1 1e-5 --lr-2 3e-6 --lr-3 1e-6 \
  --trace-lr 1e-5 --block-size 40 --seed 0
```

参数由脚本统一传给三组；独立 TraceDraft 组中的外层参数不触发外层训练。所有参数修改均使用新 `OUT`；原任务中断续跑时只需在**原命令末尾加 `--resume`**。

若研究块内长程策略，可在命令后添加：

```bash
--trace-mode block_persistent --trace-lr 1e-7 \
--replay-mib 64 --replay-positions 64 --anchor-weight 1e-4
```

这组数值是可修改的运行配置。两组带 TraceDraft 的方法都在块内延续更新，在新块重新初始化学习器、replay 和 anchor；独立组始终以初始 draft 为基础，组合组以当时的 ensemble checkpoint 为基础。它们共享分块和初始化流程；此模式不等同于跨全部块保留状态的原始 Persistent。

## 4. 数据集与分块

换数据集时修改 `QUESTIONS` 和 `OUT` 即可。每行一个完整 conversation：

```json
{"question_id": "q1", "turns": ["Question text"]}
{"question_id": "q2", "turns": ["First question", "Follow-up question"]}
```

ID 唯一，默认按文件顺序运行全部问题及全部 turn。MT-Bench 为 80 个 conversation、160 个 turn，默认 40 题一块，会在第一块后执行三个学习器的训练，第二块使用更新后的集成模型。默认最后一块不训练，因为没有后续问题使用更新。

对于 PubMedQA、MATH-500、TheoremQA、LMSYS 等，agent 先准备同样格式的 JSONL。需要随机子集时固定抽样种子，保存一份文件供三组共同使用。短实验也需至少两个块，例如 3 题应设 `--block-size 2`，否则不会执行外层训练。

多个数据集可分配给不同卡，在单独终端中使用 `CUDA_VISIBLE_DEVICES=0/1/2/3` 和独立 `OUT`。每个脚本顺序跑完本数据集三组，不需要自动监控任务。

## 5. 自动生成的结果

```text
OUT/
  comparison_plan.json
  onlinespec/
  tracedraft/
  onlinespec_trace/
    config.json                 超参数
    manifest.json               模型、数据、源码版本
    question_order.jsonl        实际顺序
    answers.jsonl               文本及实际生成 token IDs
    trace_updates.jsonl         在线 loss、漂移、更新接受/拒绝、回放信息
    summary.json                全数据集指标
    windows_100.json             每独立 100 个 conversation 的统计
    ensemble_history.json      权重、meta loss 与累计 loss
    blocks/0000/
      merge/                    本块服务 checkpoint 及合并记录
      infer/
        turns.jsonl             原始计数与时间
        adaptation_config.json  replay 字节预算 B、抽样数 m、anchor 系数 λ
        hardware.json           GPU 型号、精度、依赖版本
      learner_0/                外层训练日志和 checkpoint
      learner_1/
      learner_2/
  comparison/
    comparison.md
    comparison.csv
    comparison.json
```

三组目录布局相同，独立 TraceDraft 组没有 `learner_*` 训练目录。每个 learner 的 `osd_updates.jsonl` 记录七步损失和 optimizer 更新；这是复用训练后端的日志文件名，外层算法仍是 OnlineSPEC。`training_loss.json` 保存用于 meta 权重的汇总。

| 指标 | 定义 |
|---|---|
| `average_length` | 接受 children 总数 / 验证步数，**不含 root** |
| `average_length_including_root` | 上述值 + 1，尚未扣除末轮截断 |
| `returned_tokens_per_step` | 实际返回 token 数 / 验证步数，已扣除截断 |
| `acceptance_percent` | `100 × 接受 children / 候选路径宽度之和`，本项目统计口径 |
| `tokens_per_second_generation` | 返回 token 总数 / 生成总时间，包含 TraceDraft 更新 |
| `tokens_per_second_training_inclusive` | 另计 Reset、全部三个学习器训练/保存、合并时间 |
| `speedup_generation` / `speedup_training_inclusive` | 相应 tok/s 除以本次 **ENS-EAGLE-3** 的 tok/s |
| `tokens_per_second_pipeline` | 进一步计入模型重载、warmup、进程启动和 I/O |

主表中 `onlinespec` 为 `1.0`，其余两组的比值均以它为分母。`comparison.json` 中的 `pairwise_gains` **还提供组合组相对独立 TraceDraft 的提升**；Markdown 报告末尾也列出这两种组合增益。相对提升百分比为 `100 × (ratio − 1)`，接受率百分点差单独记录。

统计使用总计数 / 总时间。OnlineSPEC 组和组合组还会核对外层学习轨迹，确保组合组的服务更新没有写回外层。若完整输出不同，程序保存原始记录及 `comparison_failure.json`，不把它生成有效的配对加速比。

Agent 完成后提交 `comparison/`、各组 `summary.json` 和 `config.json`。checkpoint 与 Adam 状态按块保留用于续跑，结果目录应放在磁盘空间充足的位置。

## 6. 两层方法如何组合

OnlineSPEC 用已完成的一块回答训练外层学习器，再合并成后续请求的服务 draft。TraceDraft 用当前 prompt 和已验证前缀，在当前回答内更新服务 draft 的 norm/head。外层训练始终从其自己的磁盘 checkpoint 启动；临时 TraceDraft 权重不会写回外层起点。

独立 TraceDraft 组使用相同回答内更新方法，外层始终不训练。这样，一张表可以同时回答：TraceDraft 单独运行如何、OnlineSPEC 单独运行如何、二者组合能否继续改善。

算法对应见 [IMPLEMENTATION.md](IMPLEMENTATION.md)，来源和许可证见 [THIRD_PARTY.md](THIRD_PARTY.md)。原 `run_pair.py` 仍只跑两组 OnlineSPEC 对照；**本指南的三组实验请使用 `scripts/run_comparison.sh`**。无需调用测试、诊断或打包脚本。

## 7. 上传到 GitHub

在本地 `adapted-ea` 仓库根目录执行一次，即可提交两个文件夹：

```bash
git add -- osd_tracedraft onlineSpec-trace
git commit -m "Add OSD and OnlineSPEC TraceDraft experiment runners" --only -- osd_tracedraft onlineSpec-trace
git push origin HEAD
```

保留 `onlinespec_trace/core/`、配置、源码来源和许可证；权重、optimizer、缓存、本地大体积结果和 ZIP 已通过 `.gitignore` 排除。服务器 agent 只需克隆仓库并提供当地的模型、数据和结果路径。
