# EAGLE-3 + OSD：两组比较实验

本目录用于在**同一个 target、同一个匹配的 EAGLE-3 draft、同一份数据集**上比较：

| 结果目录 / `method` | 实验方法 |
|---|---|
| `osd` | EAGLE-3 + OSD |
| `osd_tracedraft` | EAGLE-3 + OSD + TraceDraft |

下面是 Linux / Bash 服务器的核心运行方式。Agent 只需从用户取得 target 路径、对应 draft 路径、数据集 JSONL、结果目录和 GPU 编号，然后运行脚本。脚本自动记录数据并生成比较报告，不需要另外启动监控或检查脚本。

## 1. 安装一次

两个实验目录应放在仓库根目录：

```text
adapted-ea/
  osd_tracedraft/
  onlineSpec-trace/
```

在另一台机器上：

```bash
git clone https://github.com/YaoWangyinjie/adapted-ea.git
cd adapted-ea
python3.12 -m venv .venv-experiments
source .venv-experiments/bin/activate
python -m pip install -r osd_tracedraft/requirements.txt
```

若已有仓库，直接进入该仓库并更新到包含这两个文件夹的版本。两个目录使用相同的依赖版本，可共用此环境。服务器需有可用的 NVIDIA CUDA 驱动；正式实验默认未量化 target、单卡运行。目录内已包含固定版本的推理/训练后端，不需要另装上游 OSD、OnlineSPEC 或整个 adapted-ea 包。

## 2. 设置路径，运行两组

以下命令均从 `adapted-ea` 仓库根目录执行，路径填写为服务器上的**绝对路径**。

```bash
export PROFILE=deepseek
export TARGET=/absolute/path/DeepSeek-R1-Distill-Llama-8B
export DRAFT=/absolute/path/matching-EAGLE3-DeepSeek-8B
export QUESTIONS=/absolute/path/mt_bench/question.jsonl
export OUT=/absolute/path/results/osd/deepseek_mtbench_seed0

CUDA_VISIBLE_DEVICES=0 bash osd_tracedraft/scripts/run_pair.sh
```

这一条命令会依次运行 `osd`、`osd_tracedraft`，然后自动统计两组结果。**不会额外运行冻结 EAGLE-3 或独立 TraceDraft。** 脚本也支持 `GPU=0` 的写法；同时设置时以 `GPU` 为准。

切换 Vicuna 时，只需更换模型配置和输出目录，再运行同一脚本：

```bash
export PROFILE=vicuna13b
export TARGET=/absolute/path/vicuna-13b-v1.3
export DRAFT=/absolute/path/matching-EAGLE3-vicuna-13b-v1.3
export OUT=/absolute/path/results/osd/vicuna_mtbench_seed0

CUDA_VISIBLE_DEVICES=1 bash osd_tracedraft/scripts/run_pair.sh
```

`deepseek` 对应 DeepSeek-R1-Distill-Llama-8B，`vicuna13b` 对应 vicuna-13b-v1.3。`DRAFT` 必须是与该 target 匹配的 **EAGLE-3** 权重；EAGLE / EAGLE-2 draft 不能替代。Vicuna 默认上下文上限为 2048，DeepSeek 为 4096，默认最多生成 1024 token。

## 3. 换数据集或实验参数

换数据集只需更换 `QUESTIONS` 和 `OUT`。JSONL 每行是一个完整 conversation：

```json
{"question_id": "q1", "turns": ["Question text"]}
{"question_id": "q2", "turns": ["First question", "Follow-up question"]}
```

`question_id` 唯一；脚本运行文件中的**所有问题及全部 turn**，默认保持文件顺序。MT-Bench 标准文件为 80 个 conversation、160 个 turn。其他数据集需先转为相同格式；需要抽样时由 agent 用固定种子生成一份子集文件，再让两组共用该文件。

默认参数如下，可直接启动：

| 参数 | 默认值 |
|---|---|
| OSD 分块 / 训练 | 40 个 conversation；2 epochs；batch size 2 |
| OSD 学习率 | `1e-5` |
| OSD 验证回退 | 开启；按完整 conversation 留出验证题 |
| TraceDraft | Reset；学习率 `1e-5`；balanced 监督 |
| 更新触发 | 当前回答第 5 次验证后尝试一次更新 |
| prompt / response 采样上限 | 各 64 个位置 |
| 解码 / seed | greedy；FP16，输出 norm 保留 FP32；seed 0 |

例如调整学习率和分块：

```bash
CUDA_VISIBLE_DEVICES=0 bash osd_tracedraft/scripts/run_pair.sh \
  --osd-lr 1e-5 --osd-block-size 40 --trace-lr 1e-5 --seed 0
```

“第 5 次验证后更新”不是训练 5 个 epoch，也不是积累 5 个问题。一个回答最多尝试一次 TraceDraft 更新；短回答没有到达触发位置会跳过。`--trace-update-round` 调整触发时点。

需要块内 Persistent + R + A 时，使用新 `OUT`：

```bash
CUDA_VISIBLE_DEVICES=0 bash osd_tracedraft/scripts/run_pair.sh \
  --trace-mode block_persistent --trace-lr 1e-7 \
  --replay-mib 64 --replay-positions 64 --anchor-weight 1e-4
```

以上 R/A 数值为可修改的运行配置，不代表最优参数。Reset 在每个新 conversation 恢复到当前 OSD checkpoint，同一 conversation 的后续 turn 继承临时更新。`block_persistent` 在当前 OSD 块内继承更新；块间重新载入 OSD 权重，同时清空回放并重建 anchor。

默认不训练最后一个数据块，因为没有后续问题使用其权重。数据量小于等于 block size 时不会发生 OSD 训练；短实验至少分成两个块，例如 3 题用 `--osd-block-size 2`。

同一运行因中断而续跑时，在**原命令后加 `--resume`**。改变任何参数或输入后用新的 `OUT`。

## 4. 结果位置与口径

```text
OUT/
  osd/
  osd_tracedraft/
    config.json                 完整超参数
    manifest.json               模型、数据和代码版本
    question_order.jsonl        实际问题顺序
    summary.json                总体指标
    windows_100.json             每独立 100 个 conversation 的统计
    blocks/0000/infer/
      turns.jsonl               逐 turn 原始计数、时间、截断信息
      answers.jsonl             文本及实际生成 token IDs
      training_records.jsonl    外层蒸馏使用的 token 和 mask
      trace_updates.jsonl       在线 loss、更新状态、漂移、回放信息
      adaptation_config.json    replay 字节预算 B、抽样数 m、anchor 系数 λ
      hardware.json             GPU 型号、精度和依赖版本
    blocks/0000/train/
      osd_updates.jsonl         外层优化记录
      osd_guard.json            外层验证与更新是否通过
      checkpoint/               外层权重和 optimizer 状态
  comparison/
    comparison.md               可直接阅读的比较表
    comparison.csv              表格数据
    comparison.json             完整指标与提升比例
```

`osd/` 与 `osd_tracedraft/` 使用相同目录结构。逐块控制台输出在各阶段的 `console.log`。

| 报告指标 | 定义 |
|---|---|
| `average_length` | 接受的 draft children 总数 / 验证步数，**不含 root** |
| `average_length_including_root` | 上述接受长度 + 1，尚未扣除末轮截断 |
| `returned_tokens_per_step` | 实际返回 token 数 / 验证步数，已扣除截断 |
| `acceptance_percent` | `100 × 接受 children / 候选路径宽度之和`，沿用本项目口径 |
| `tokens_per_second_generation` | 总返回 token / 总生成时间，包含 TraceDraft 在线更新 |
| `tokens_per_second_training_inclusive` | 另计 Reset 和 OSD 训练、保存 checkpoint 的时间 |
| `speedup_generation` / `speedup_training_inclusive` | 相应 tok/s 除以本次 **OSD 组**的 tok/s |
| `tokens_per_second_pipeline` | 进一步计入模型重载、进程启动、warmup 和 I/O |

OSD 组加速比为 `1.0`；组合组 `1.05` 表示相对 OSD 吞吐提升 5%。JSON 同时保存相对接受率提升、接受率百分点差、接受长度提升。所有汇总均从总数/总时间计算。两组输出不一致时，报告会停止计算加速比并保留原始记录。

Agent 完成后向用户提交 `comparison/` 及两组 `summary.json`、`config.json`；原始日志和 checkpoint 保留在指定 `OUT` 内。

## 5. 实现定位与上传

OSD 在完成一块回答后进行七步递归蒸馏，更新 draft 主干和输出层；组合组额外在当前回答生成期间更新 norm/head。外层训练从该块开始时的 OSD checkpoint 重新载入，服务时产生的 TraceDraft 临时权重不会写回外层训练起点。

这是基于 OnlineSPEC 单学习器管线的 **OSD–EAGLE-3 适配**，默认增加验证回退，两组使用同样设置；并非原始 OSD 仓库直接支持 EAGLE-3。详细参数见 [IMPLEMENTATION.md](IMPLEMENTATION.md)，代码来源和许可证见 [THIRD_PARTY.md](THIRD_PARTY.md)。三组 ensemble 实验使用另一个目录的 [README](../onlineSpec-trace/README.md)。

两个文件夹可以直接提交到当前 `adapted-ea` 仓库，在仓库根目录运行：

```bash
git add -- osd_tracedraft onlineSpec-trace
git commit -m "Add OSD and OnlineSPEC TraceDraft experiment runners" --only -- osd_tracedraft onlineSpec-trace
git push origin HEAD
```

`.gitignore` 已排除模型权重、optimizer、缓存、本地大体积运行目录及 ZIP。保留 `backend/`、配置和许可证；无需上传 `D:/model_weights`、外部 OSD / OnlineSPEC 克隆目录或虚拟环境。多个实验可分别设置 `CUDA_VISIBLE_DEVICES=0/1/2/3` 运行，每组使用不同 `OUT`，每个进程只占一张卡。
