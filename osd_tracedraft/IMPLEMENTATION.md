# OSD / TraceDraft 实现说明与详细参数

本目录运行四个配对组：

| 命令中的 method | 推理和更新 |
|---|---|
| baseline | 冻结的 EAGLE-3 |
| tracedraft | EAGLE-3 + TraceDraft |
| osd | EAGLE-3 + OnlineSPEC 式 OSD 块间蒸馏 |
| osd_tracedraft | 相同的 OSD 外层训练 + 当前回答内的 TraceDraft 更新 |

这里的 OSD-EAGLE-3 是针对 EAGLE-3 特征预测结构的适配。原始 OSD 的独立小语言模型不能直接充当 EAGLE-3 draft。实现参考 OnlineSPEC 的单模型 pipeline_eagle3.py，不是它的多模型 hedge/ensemble 管线。

运行两组对比请使用 [README.md](README.md) 中的新入口；下文保留实现细节和进阶用法。

## 1. 更新顺序

每个 OSD 数据块默认包含 40 个完整 conversation。MT-Bench 的两个 turn 属于同一个 conversation，不在 turn 中间切块。

1. 从当前 OSD checkpoint 载入 draft，生成该块的全部回答。
2. 组合组在当前回答生成时调用本项目 TraceDraft；默认每个新 conversation 恢复到当前 OSD checkpoint，第二个 turn 继承第一个 turn 的临时更新。
3. 保存实际输入/输出 token IDs 及本 turn 的 assistant mask。
4. 独立训练进程重新载入该块开始时的 OSD checkpoint，使用该块已生成的数据训练；绝不从最后一个问题的临时 TraceDraft head 开始训练。
5. 下一块载入新的 OSD checkpoint。默认不训练最后一块，因为没有后续问题受益；需要产出最终权重可传 --train-last-block。

OSD 训练冻结 target 和 token embedding，更新 draft 的 fc、midlayer、norm、lm_head。使用七步递归预测，soft cross entropy 的第 r 项乘 0.8^r；AdamW 状态跨块延续。训练主权重保持 FP32，推理按配置转换为 FP16/BF16；最终输出 norm 默认保留 FP32，激活转回 head 的推理精度。优化器 batch 默认 2，通过逐序列梯度累积实现，避免一次保存多条长序列的计算图。

训练 attention 保留原始因果前缀，并逐轮增加同一位置的递归状态，不能用七次普通 causal forward 代替。每个样本的归一化分母按本优化器 batch 的最大有效序列长度和样本数计算。teacher argmax 不在 draft 的缩减词表内时，该位置不计入损失。

与上游脚本的明确差别：使用真实生成 token，监督当前 turn 新产生的 assistant token；不重新解析聊天字符串，不重建 d2t/t2d，不把多轮对话的旧答案再次计入当前 turn 的监督。使用本项目的推理器和统一上下文/树参数。因此应称为本项目下的 OSD-EAGLE-3 适配，不应声称逐项复现 OnlineSPEC 论文数字。

### Reset 与块内延续

默认 --trace-mode reset。

可选择 --trace-mode block_persistent，在同一个 OSD 块内跨 conversation 延续 TraceDraft head、Adam 状态和回放。块间 OSD 改变 draft 主干，所以必须清空旧特征回放并重建学习器；新的 anchor 是更新后的 OSD checkpoint。该模式不是跨全部数据块、固定主干的原始 Persistent，名称已明确区分。

R/A 参数示例：

    --trace-mode block_persistent --replay-mib 64 --replay-positions 64 --anchor-weight 1e-4

三者是可调实验参数，示例值不是经过验证的最优值。Replay 预算记录为字节，抽样位置数和实际占用分别记录；anchor 是各参数相对初始值的平方距离正则。

TraceDraft Reset 默认学习率为 1e-5，便于延续已有的 Reset 设置；它不是对组合实验最优值的预判。默认在当前回答的第 5 次验证完成后更新一次；--trace-update-round 改变触发位置，--trace-every 改变 generation call 的更新间隔。一个回答最多触发一次。这里不是训练 5 轮。未达到触发位置的短回答记录为 warmup_not_reached。

## 2. 目录与环境

将 osd_tracedraft 整个文件夹复制到服务器的 adapted-ea 下即可。backend/ 已固定打包本次测试所用的 EAGLE-3 / TraceDraft 后端，不依赖服务器上可能不同版本的 eagle/model/。SNAPSHOT.json 记录每个源文件的 SHA256，原始文件与许可证一并保留。不需要把 OSD 或 OnlineSPEC 加进 PYTHONPATH。数据集与模型权重由你提供。

所有命令在包含 eagle/、osd_tracedraft/ 的 adapted-ea 目录运行：

    cd /your/path/adapted-ea
    python -m venv .venv-osd
    source .venv-osd/bin/activate
    python -m pip install -r osd_tracedraft/requirements.txt

requirements 使用 torch 2.6.0、transformers 4.53.1；按服务器驱动选择兼容的 CUDA PyTorch wheel。不要把本地 CPU 测试环境复制到服务器。

每个进程只允许看到一张 CUDA 卡。默认使用 FP16；改用 BF16 时，四个配对组必须统一。正式运行会记录 CUDA/PyTorch/Transformers 版本、卡名、显存总量、权重 SHA256、源码 SHA256、数据 SHA256、题目顺序和全部参数。

### 模型

- deepseek：DeepSeek-R1-Distill-Llama-8B；使用其 tokenizer 自带 chat template，默认上下文 4096。
- vicuna13b：vicuna-13b-v1.3；使用 FastChat vicuna_v1.1 格式，默认上下文 2048。

两者都必须提供与 target 匹配的 EAGLE-3 draft。Vicuna 的 EAGLE/EAGLE-2 checkpoint 不能拿来替代 EAGLE-3。加载器检查词表映射、三层特征投影、目标模型尺寸，并拒绝混放 .bin 与 .safetensors 的不明确目录。结构检查无法替代 checkpoint 来源核验；请保留下载来源。

默认生成最多 1024 token，提前给回答保留上下文预算。超长 prompt 明确从左侧截断并保留 BOS，逐 turn 记录截断数量。可使用 --prompt-truncation error 禁止截断。所有比较组使用同一策略；Vicuna 的 2048 上限不会被静默扩展。长文摘要数据要检查 prompt_truncated_turns。

## 3. 先做小规模运行

先从真实 question.jsonl 取少量完整问题写成独立 smoke.jsonl，例如 4 题。为触发一次 OSD 更新，smoke 设置 block_size=2；只有一个数据块时默认不会训练。

    CUDA_VISIBLE_DEVICES=0 python -m osd_tracedraft.run \
      --model-profile deepseek --method osd_tracedraft \
      --target /path/to/DeepSeek-R1-Distill-Llama-8B \
      --draft /path/to/matching-eagle3 \
      --questions /path/to/smoke.jsonl --output /results/smoke/combined \
      --osd-block-size 2 --osd-epochs 1 --max-new-tokens 64 \
      --verify-greedy 4 --verify-tokens 32

--verify-greedy N 对前 N 个 conversation 的每个 turn 运行短 target-only greedy 检查，检测输出前缀一致性。这部分开销单独记录，不计入 generation tok/s，但计入完整 pipeline 时间。正式四组应设置相同值。小规模检查后再跑全量。

## 4. 两种模型分别运行

脚本会依次跑四组，最后生成配对结果。权重路径采用环境变量，不猜测服务器文件名：

    export TARGET=/your/path/new-eagle/model_weight/DeepSeek-R1-Distill-Llama-8B
    export DRAFT=/your/path/new-eagle/model_weight/its-matching-EAGLE3
    export QUESTIONS=/your/path/adapted-ea/eagle/data/mt_bench/question.jsonl
    export OUT=/your/results/deepseek_mtbench
    GPU=0 bash osd_tracedraft/scripts/run_deepseek.sh --verify-greedy 2

Vicuna：

    export TARGET=/your/path/new-eagle/model_weight/vicuna-13b-v1.3
    export DRAFT=/your/path/new-eagle/model_weight/its-matching-vicuna-EAGLE3
    export OUT=/your/results/vicuna_mtbench
    GPU=1 bash osd_tracedraft/scripts/run_vicuna13b.sh --verify-greedy 2

--trace-lr 1e-5、--osd-lr 1e-4 等参数可以接在脚本后面。修改任何参数要用新的 OUT。四组都会保存该配置，其中没有开启的更新参数不会生效。

默认 OSD 使用 lr=1e-5、epochs=2、batch=2、chunk=40，并开启整块更新的验证回退。旧示例中的 1e-3 在本地小规模测试中出现过更新后接受长度大幅下降，因此不再作为默认值。每次比较组合增益时，OSD 和 OSD+TraceDraft 的 OSD 参数必须完全一致。

### 更新验证与诊断开销

TraceDraft 默认从本次采样位置留出验证集，比较实际推理 norm/head 在候选更新前后的 KL。只有 `after <= before + atol + rtol * abs(before)` 才提交更新；默认 atol=1e-6、rtol=0。未通过时恢复权重、Adam 状态、版本号、回放和抽样 RNG；没有足够验证位置时跳过。对应参数为 `--trace-validation-atol`、`--trace-validation-rtol`。

默认 `--trace-diagnostics basic` 保留更新前后的推理精度验证、梯度和单步漂移，省去完整诊断中的额外训练/验证前向及大 head 的全量变化统计。大 head 的 `inference_changed_fraction_sampled` 是最多 4096 个位置的估计；精确字段此时为 null。需要完整 loss 和精确 head 变化比例时使用 `--trace-diagnostics full --trace-diagnostics-every 4`，每四次更新尝试记录一次完整诊断。norm 的变化比例仍精确记录。

OSD 默认在当前块按完整 conversation 留出最多 2 题，同题的所有 turn 一起留出，至少保留一题训练；每条验证序列最多抽取 128 个有效位置，使用同一组位置比较更新前后 FP32 draft 的七步加权 CE。对应参数为 `--osd-validation-questions`、`--osd-validation-positions`、`--osd-validation-atol`、`--osd-validation-rtol`。没有可用验证集、非有限数值或验证 CE 上升时，下一块继续使用父 checkpoint 及其原 optimizer。失败尝试和验证开销照常计时。该策略改变了训练协议，是加入验证保护的 OSD-EAGLE-3 适配，不能标为原论文基线的原样复现。

最终 norm 的 FP32 权重从 checkpoint 直接恢复，避免小更新先被 FP16 转换舍入掉；其它 draft 层及 head 仍按配置使用推理精度。所有配对组统一此选项，报告拒绝比较 norm 精度不同的组。

如需单独研究原更新设置，可以显式传入 `--osd-lr 1e-3 --no-osd-guard --no-trace-validation-guard --trace-diagnostics full --no-output-norm-fp32`，并使用新输出目录。这些开关恢复相关策略设置，不代替历史版本的源码快照。验证 loss 改善仍需用接受长度和总耗时检验，不能直接换算成加速收益。

## 5. 四卡连续调度

复制 configs/suite.example.json，填入两个模型及其 draft 路径、全部数据集路径、output_root。所有相对路径相对于启动时的 adapted-ea 目录。删除暂时不跑的数据集或模型条目。

    python -m osd_tracedraft.queue --config /path/to/my_suite.json --gpus 0,1,2,3 --dry-run
    SUITE=/path/to/my_suite.json bash osd_tracedraft/scripts/run_four_gpus.sh

默认 2 个模型 × 7 个数据集 × 4 个方法 × 1 个 seed = 56 个独立任务。队列按实际问题数从小到大排序；每张卡完成一个任务就领取下一个任务。它通过操作系统等待子进程退出，无需 agent 自动监控，也不消耗额外模型 token。每个 worker 显式设置 CUDA_VISIBLE_DEVICES 为单个物理卡编号；模型内部只用 cuda:0。

任务失败时保留日志，该卡继续下一任务；最后汇总失败清单，不把失败任务当作零分。status.json 保存正在运行的 PID，completion.json 保存退出码及汇总错误。

本地自带的 pubmed_qa/gsm8k 等文件可能只有 80 题，不应误写成 500。配置中的 expected_questions 会验证数量。所有数据统一为 JSONL：

    {"question_id": "unique-id", "turns": ["question text"]}
    {"question_id": "another-id", "turns": ["first turn", "second turn"]}

question_id 必须唯一；数据集实际有几题就运行几题，没有默认只跑前 80 题的隐藏截断。

## 6. 结果文件

    output_root/
      deepseek/mt_bench/seed_0/
        baseline/
        tracedraft/
        osd/
        osd_tracedraft/
        comparison/
          comparison.md
          comparison.csv
          comparison.json
      vicuna13b/...
      queue/
        plan.json
        status.json
        completion.json
        *.log

每个方法目录包含：

- config.json / manifest.json / question_order.jsonl：全部参数、文件指纹和顺序。
- blocks/0000/infer/turns.jsonl：逐 turn 原始计数、推理用时、截断、更新状态、显存峰值。
- answers.jsonl：文本答案、实际生成 token IDs、输出 hash、greedy 检查 token。
- training_records.jsonl：用于 OSD 的准确 token IDs 与 mask。
- trace_updates.jsonl：实际推理验证 loss、参数单步漂移、拒绝原因、replay 实际占用及配置；完整诊断额外记录训练 loss 和精确 head 变化比例，basic 模式明确记录抽样比例。
- adaptation_config.json：包括 replay_bytes、replay_positions、anchor_weight 的完整在线配置。
- blocks/0000/train/osd_updates.jsonl：每次 optimizer step、七轮 CE、覆盖位置数、梯度范数、实际学习率。
- blocks/0000/train/osd_validation_split.json / osd_guard.json：训练/验证题目 ID、验证前后 CE、阈值、更新是否通过及拒绝原因。
- blocks/0000/train/checkpoint/：通过验证的新 draft 或回退的父 draft、完整原 config、optimizer.pt（如有）、provenance.json。
- summary.json：全数据集汇总；block_summaries.json：每个 OSD 块；windows_100.json：每独立 100 个 conversation 的生成阶段统计。
- console.log / failure.json：错误与运行输出。

参数漂移现沿用原 OnlineHead 记录；anchor_relative_drift 在原实现未启用硬漂移阈值时可能为 null，不能解读为零。A 的软约束系数仍准确记录并生效。

### 指标公式

设 A 为接受的 draft children 总数，S 为验证步数，W 为候选路径宽度之和，N 为实际返回 token 数。

- average_length = A/S：不含 root，与本项目 children/step 对齐。
- average_length_including_root = (A+S)/S：验证计数口径，含 root。
- returned_tokens_per_step = N/S：扣除了末轮 EOS/长度截断的实际返回长度。
- discarded_verified_tokens = A+S-N：截断差额。
- acceptance_percent = 100 A/W：沿用本地 EA4。W 中候选路径宽度包含 root；不要把它误称为所有树节点的接受比例或经典 rejection sampling 的逐 token 概率。
- tokens_per_second_generation = N / generation_seconds：包括本轮 TraceDraft 数据收集和在线更新。
- tokens_per_second_training_inclusive = N / (generation_seconds + reset_seconds + OSD training stage seconds)：额外计入 OSD 训练及 checkpoint 保存。
- tokens_per_second_pipeline：计入进程启动、模型重载、warmup、tokenization、数据文件 I/O 和 correctness probe 的整个运行时间。

speedup_* 均为对应 tok/s 除以本次、同配置、冻结 EAGLE-3 baseline 的 tok/s；这不是相对原始自回归解码的倍数。报告同时计算 OSD+TraceDraft 相对 OSD 的增益。comparison.json 的 slow_checkpoint_comparison 还会核对两组每次 OSD 更新后保存的外层权重是否一致。

采用总数/总时间，不平均逐题或逐块百分比。报告拒绝比较不同 checkpoint、数据顺序、树设置或输出 token 序列。多个 seed 应分别保留配对结果；不以最优 seed 代替总体结果。

每 100 题的窗口只报告生成阶段吞吐。OSD 开销按真实块记录并计入整次运行，不任意摊到跨块的窗口。

## 7. 续跑与测试

    python -m osd_tracedraft.run --config /results/.../config.json --resume
    SUITE=/path/to/my_suite.json bash osd_tracedraft/scripts/run_four_gpus.sh --resume
    python -m pytest osd_tracedraft/tests -q

只从已经完成的 OSD 块续跑；中途失败的块完整重跑，旧目录以 .failed-时间戳保留。不会把不完整 JSONL 直接拼接成结果。断电遗留 RUNNING.lock 时，先确认该任务 PID 已退出再手动移除该锁。

同一个结果目录不能并发运行两个任务。原始权重、历史 final_results 和论文文件均不被修改。当前测试范围见 VALIDATION.md；更换模型或服务器环境后仍需完成上面的 smoke 检查。

## 本地 8GB 显存功能检查

可选参数 --target-quantization nf4 --draft-device cpu --cpu-threads 4 用于 GPU target / CPU draft 的功能测试；默认仍是原来的全 GPU、未量化路径。NF4 需要额外安装 bitsandbytes，本机测试使用 0.48.2 和 Accelerate 1.10.1。运行配置和算法说明见 [local_smoke_4060/README.md](local_smoke_4060/README.md)。

仅比较 osd 与 osd_tracedraft 时，报告命令添加 --reference-method osd；默认仍以冻结 EAGLE-3 为参考。量化方式、draft 设备、CPU 线程数也进入匹配检查，避免混用不同执行设置的吞吐。

本次代码更新与真实 8B 验证见 [修复验证报告](local_smoke_4060/fix_validation/README.md)。新增统计包括 OSD 的已提交步数、尝试步数和拒绝块数，以及 TraceDraft 的验证拒绝 turn 数与跳过 turn 数。
