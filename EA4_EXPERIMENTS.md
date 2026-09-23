# EA4：在线蒸馏实现与实验方案

这是一套用于验证在线适应能否带来**实际解码收益**的实现。默认选择 `persistent + response + end + FP32 master`，优先修复此前诊断的问题，并提供可单独控制的消融开关。默认超参数是实验起点，尚未在真实 DeepSeek 8B 权重上验证为最优。

EA2、EA3、旧评测入口及已有结果均未修改。EA4 不继承 EA2，避免把旧的 prefill、标签构造、参数转换、缓存恢复路径带进新实验。

## 1. 文件与适用范围

| 文件 | 用途 |
|---|---|
| `eagle/model/ea_model_4.py` | target prefill、树验证、连续训练轨迹、更新时机和缓存管理 |
| `eagle/model/online_head.py` | FP32 参数、事务式 Adam、来源均衡、验证集、replay、漂移限制 |
| `eagle/evaluation/gen_ea_answer_ea4.py` | 单进程按问题顺序评测、reset/persistent、冻结阶段、计时与可复核记录 |
| `eagle/evaluation/plan_ea4_experiments.py` | 生成实验矩阵及 Bash/PowerShell 命令，不自动启动实验 |
| `eagle/evaluation/compare_ea4.py` | 按 question/turn 配对，检查输出一致性和实验设置，比较吞吐及接受情况 |
| `tests/test_ea4.py` | 无下载、无 GPU 的真实 PyTorch / 小型 Llama / EAGLE 测试 |

当前支持本仓库的 **Llama 架构 target + 对应 EAGLE-3 draft、batch=1、greedy decoding**。与你正在使用的 DeepSeek-R1-Distill-Llama-8B 架构相符；仍须用实际匹配的 draft 做下面的 smoke 验证。Qwen、采样、量化权重、CPU/disk offload、大模型多卡运行没有验证。多卡代码按实际设备分组缓存，但不要把这一点当作多卡性能保证。

只开放 `head_only`（norm + lm_head）、`norm_only`、`lm_head_only`。fc、midlayer 和 target 始终冻结，因此缓存与 replay 特征不会因更新 backbone 而失效。没有实现 EAGLE-3 原始训练的多步自回归 rollout loss，也没有声称单步 KD 足以改善树的所有深度。

## 2. 实际训练的是什么

设 token 为 `x_0 ... x_(L-1)`，target 消费 `x_j` 后得到三层拼接特征 `H_j` 与下一 token 的 logits `z_j`。正确配对是：

```text
draft 第 j-1 个位置输入：H_(j-1), x_j
draft 第 j-1 个位置输出：预测 x_(j+1) 的 logits
teacher 标签：z_j                         j = 1 ... L-1
```

因此训练时使用 `hidden[:, :-1] + input_ids[:, 1:]`，对应 `teacher_logits[:, 1:]`。`--alignment legacy` 故意恢复一个位置的错位，作为负对照；它只隔离错位因素，不是完整复刻 EA2 的全部行为。边界处无法配对的位置会被舍弃，负对照样本数可能略少，应查看日志。

对缩减词表 draft，先用 checkpoint 的 `t2d/d2t` 验证映射，再取 draft 词表内的 teacher logits 重新归一化。完整 teacher 的 argmax 不在 draft 词表内的位置不参与 KD，沿用本项目 EAGLE 训练的筛选思路。日志提供 `valid_positions / selected_positions` 和 `mean_draft_vocab_mass`，用于判断模型或数据是否存在词表覆盖瓶颈。这里的 KL 是**词表子集条件分布**的 KL，不是整个 target 词表的 KL。

teacher 为 stop-gradient；draft loss 为温度缩放后的 `T² KL(p_teacher || p_draft)`。FP32 norm/head 主参数保留跨更新的微小增量，Adam 的一阶、二阶矩同样保留。通过有限值和漂移检查后才同步到推理 dtype。`--precision roundtrip` 故意在每次 optimizer step 后将主参数往返转换到推理 dtype，用于验证更新丢失。

原始设想“在每个 question 的 prefill 上训练，然后再生成”，对应：

```text
--update-at prefill --source prompt --weight-mode persistent
```

它在同一次 target prefill 的结果上训练，不额外跑一次 target prefill。若希望包含最后一个 prompt 位置对首个 response token 的预测，可用 `--source balanced`；`prompt` 严格只包括预测 prompt 内 token 的位置。默认 `end + response` 则在回答完成后训练，用于后续 turn/question；**不会改善刚刚已经生成完的这段答案**。

## 3. 修复和保护的边界

- **三层 hidden 固定取法**：明确取本地 KV 模型返回的前三个选定层特征。即使额外返回 final norm，也不会误拿最后三个。
- **连续上下文**：树验证中 root 和 accepted children 都进入轨迹；即使某一步接受了零个 child，也保留 root。训练特征分块前向，临时 KV 保留完整前缀，不把不相邻片段拼成一句话；不会改写生成用的 `stable_kv`。
- **prompt/response 区分**：最后一个 prompt 位置的预测归为 response。`balanced` 先分别求均值，再按 `prompt_weight` 混合，防止长 prompt 按 token 数压倒 response。
- **有限采样**：每个来源独立、均匀 reservoir 采样；默认每来源最多 128 个位置。采样 RNG 与模型 RNG 分离。response-only 直接复用 prefill 最后一个位置的 logits，不为整个 prompt 额外计算 dense logits。
- **缓存与树**：target prefill 一次；norm/head 更新不会改变已有 target/draft KV。若在生成中更新，下一棵树在更新之后构建。不同 question/turn 不复用前缀 KV。
- **persistent/reset**：persistent 同时保留主参数、Adam 状态和 replay；reset 在 question 开始时全部还原，question 内多轮仍连续适应。`update_every` 按全局 generation turn 计数，reset 不重置调度计数。
- **回滚**：候选更新不覆盖旧 Adam 状态；异常或 guard 拒绝会恢复参数、Adam、版本、replay 和更新用 RNG。运行时异常随后上抛并停止实验，避免悄悄继续无效轨迹。
- **guard**：`max_step_drift` 限制每个训练张量的一步相对 L2 距离；`max_anchor_drift` 限制相对初始值的累计距离。它们只能限制参数漂移，不能保证接受率不下降。
- **anchor/replay**：默认关闭，单独做消融。anchor 为各张量相对初始范数的平方距离之和，量纲与 EA3 原先的大参数均值版本不同，旧 λ 不宜直接套用。replay 保存 CPU 上的冻结特征和压缩词表 logits，以张量字节数限额 FIFO 淘汰；Python 对象开销不在该限额内。

每个来源至少 5 个有效位置时，会按 `validation_fraction` 留出本次局部验证位置，本次留出位置不加入 replay。它们只用于观察拟合与量化影响，**不是独立 task 测试集，也不是自动接受/拒绝更新的依据**。多轮对话中的重复文本还可能在以后再次出现。最终泛化应使用下面的冻结后缀实验。

## 4. 环境与最短运行方式

从 `adapted-ea` 根目录运行。EA4 不依赖 FastChat/Ray/W&B。测试环境为 Python 3.12、PyTorch 2.6.0 CPU、Transformers 4.53.1、Accelerate 1.15.0。GPU 上请安装与 CUDA 匹配的 PyTorch，保留仓库兼容的 Transformers 版本；不要把 CPU 测试环境直接用于性能实验。

```bash
python -m unittest discover -s tests -p test_ea4.py -v
```

以下命令中的 `/models/target` 和 `/models/eagle3-draft` 是占位符，需替换为服务器上的实际匹配权重路径。各次运行必须使用新的输出目录；代码拒绝混入旧结果。

先生成仅 4 个 question 的 smoke 计划。它包含 no-update 与默认 adaptation，均对前 32 个生成 token 做 target greedy 参考校验：

```bash
python -m eagle.evaluation.plan_ea4_experiments --base-model-path /models/target --ea-model-path /models/eagle3-draft --question-file eagle/data/gsm8k/question.jsonl --output-root runs/ea4_smoke --plan-dir plans/ea4_smoke --suite smoke --max-new-tokens 128
bash plans/ea4_smoke/run.sh
```

Windows 可运行生成的 `run.ps1`。计划生成器只写文件，不加载模型，不启动训练。smoke 验证失败应先定位输出差异，不能用不一致的答案做加速结论。CPU 小模型测试只排除了已覆盖的逻辑错误；真实 GPU 的数值差异、checkpoint、长上下文仍需这一步验证。

正式默认方案与同条件 baseline：

```bash
python -m eagle.evaluation.gen_ea_answer_ea4 --base-model-path /models/target --ea-model-path /models/eagle3-draft --question-file eagle/data/gsm8k/question.jsonl --output-dir runs/gsm_baseline --no-update
python -m eagle.evaluation.gen_ea_answer_ea4 --base-model-path /models/target --ea-model-path /models/eagle3-draft --question-file eagle/data/gsm8k/question.jsonl --output-dir runs/gsm_ea4 --weight-mode persistent --source response --update-at end
python -m eagle.evaluation.compare_ea4 --baseline runs/gsm_baseline --candidate runs/gsm_ea4 --output runs/gsm_comparison.json
```

每次默认先跑 2 次不更新权重、不计时的短 inference warmup，随后将更新调度归零。首次 Adam 分配和首次训练 kernel 成本仍计入真实更新成本。正式计时关闭 `--verify-greedy` 和 `--profile`；它们用于独立诊断。不要让参考前向给某组额外预热后再与未参考校验的组比较。

显存方面，32k×4096 的 FP32 head 约 500 MiB，Adam moments 还需约 1 GiB，梯度、事务候选参数和候选 moments 会进一步增加峰值；加上 target/draft，不能保证 24 GiB GPU 容纳。日志记录每卡 peak allocated。优先用小样本实测，显存不足可选择 `norm_only` 或降低上下文/feature chunk；不同 scope 必须视为不同实验，不能与 head_only 混报。连续 hidden 轨迹也会占用与上下文长度成正比的显存。

## 5. 按顺序做实验，避免一次改变所有因素

先在 GSM8K、PubMedQA 和 MT-Bench 现有数据上跑，随后保留 PubMed/SUM 等长 prompt 任务作为对照。不要只挑已有正收益的 science/reasoning 子集；MT-Bench 按 category 分析，同时报告实际 token 量和截断比例。

| 阶段 | 对照 | 要回答的问题 |
|---|---|---|
| 0 正确性 | no-update / EA4，短序列 target 参考校验 | 加入学习后是否仍输出相同 greedy token？ |
| 1 标签 | `correct` / `legacy`，其余相同 | 错位是否抑制学习或损害后续接受？ |
| 2 精度 | `master_fp32` / `roundtrip`，均用 fp16 推理 | 微小梯度是否只在 loss 上可见，没有累积到推理权重？ |
| 3 数据来源 | end 的 prompt / response / balanced | 领域收益是否来自输出分布，而非重复拟合长输入？ |
| 4 更新时机 | prefill prompt、warmup balanced、end response | 当前请求收益与跨请求收益、开销分别有多大？此组同时改变来源，不能单独归因于时机；要隔离时机，应固定 source 再补跑。 |
| 5 持续性 | prefill 的 reset/persistent；warmup 的 reset/persistent | 在相同更新时间和数据来源下，跨 question 积累是否有用？ |
| 6 成本 | end response 每 1 / 8 / 32 个 turn 更新 | 即使接受率略升，是否有更新频率可带来净吞吐收益？ |
| 7 稳定性 | 默认分别加 replay、累计 guard、anchor；另扫 LR | 哪种措施确实改善后期曲线，而不是额外成本或从不触发？ |
| 8 泛化 | 前 N 个 question 适应，后缀固定权重 | 脱离继续学习后，收益能否在未用于调参的后缀维持？ |

生成阶段 1–6 的矩阵：

```bash
python -m eagle.evaluation.plan_ea4_experiments --base-model-path /models/target --ea-model-path /models/eagle3-draft --question-file eagle/data/gsm8k/question.jsonl --output-root runs/gsm_core --plan-dir plans/gsm_core --suite core --order-seeds 0,1,2
```

`core` 每个顺序 12 组；`stability` 每个顺序 10 组；`all` 去重后每个顺序 20 组。命令顺序确定，正式复测时应轮换各方法执行顺序，避免机器负载/温度的时间漂移。每条命令重新加载初始 checkpoint，不能把不同配置串在同一个已经适应的模型对象上。

`stability` 提供 LR `3e-7 / 1e-6 / 3e-6 / 1e-5`、norm_only、64 MiB replay、相对累计距离 .02 的 guard、anchor .01、默认前 32 个 question 后冻结。它们是待检验候选值。调参只用开发前缀，选好配置后再跑独立后缀；不要用同一后缀反复筛选 LR。

来源实验中 prompt/response 各上限 128；balanced 为 64+64，总上限同为 128。短响应、词表过滤和 holdout 会使实际训练位置数不同，必须同时比较 `train_positions`、来源数量和开销；equal cap 不等于 exact token matching。若需要严格数量匹配，应基于日志选择可满足固定有效样本数的子集后复测。

`end + reset` 对单轮 question 不会把更新用于本题生成或下一题，因而不作为 persistent 的主要对照。多轮 question 中它仍可能影响后续 turn。`--freeze-after N` 的边界按选取并打乱后的 question 数量计算，`N` 应小于实际 question 数；冻结后的模型保持前缀学习结果，停止收集和训练。

```bash
python -m eagle.evaluation.compare_ea4 --baseline runs/gsm_core/baseline_order0 --candidate runs/gsm_stability/end_response_freeze_order0 --phase frozen
```

baseline 必须覆盖同样整个 question 顺序；比较器会用候选冻结后缀的 question/turn 去配对 baseline 后缀。此比较不计前缀训练成本，只回答可迁移收益；整体在线效率要看 whole-stream 结果。

顺序实验采用至少 3 个 `order-seed`，每个顺序内部配对 baseline，报告各顺序结果范围；同一顺序再做独立重复计时。persistent 的 question 存在依赖，不能把每个 question 当作 IID 样本来夸大显著性。

模型因素需要另选一组 **Llama target + 与它严格匹配的 EAGLE-3 draft**，在相同数据、生成预算和各自 baseline 上重复实验。不能换 target 却复用 DeepSeek 的 draft，也不能只横比两组绝对接受率。比较各自提升量、词表覆盖、初始一致率、响应长度及每次更新成本，才能判断模型匹配是否影响收益。

## 6. 看哪些指标，如何判读

每次生成输出 `stats.jsonl`，每个完成的 turn 立即 flush；`answers.jsonl` 保存文本，stats 中另存生成 token ID 的 SHA256。`manifest.json` 保存参数、question 顺序、数据/源码 hash、软件/GPU、层索引、本地 config hash；可用 `--hash-weights` 额外哈希全部本地权重。远程权重应固定 revision 或先下载到固定目录，路径名称相同不自动证明权重内容相同。

- **首要指标**：`sum(new_tokens) / sum(generation_seconds)`。计时从生成调用内开始，包含 teacher 收集、feature forward、训练、更新、回答结束后的 adaptation 和同步；不包括模型加载、chat template/tokenization、答案 decode/写盘。不能把它称为服务全链路延迟。
- **解码收益**：`returned_tokens_per_step`；同时记录 `accepted_children_per_step`、各 depth 的 `survival / trials`。raw accepted children 在 EOS/长度截断前统计，而 returned tokens 是实际返回量，两者不能混用。
- **兼容旧结果的指标**：`legacy_path_width_acceptance` 分母是各步最长候选路径宽度之和，包含 root 宽度，**不是整棵树所有 draft token 的接受率**。整树节点另存 `verified_tree_nodes`。
- **是否真的更新**：`master_delta_norm`、`inference_delta_norm`、`inference_changed_fraction`、`cast_error_norm`。微小主参数更新暂时不改变 fp16 权重并不必然错误，需看跨 turn 是否最终累积到可表示变化。
- **是否学对东西**：train/validation 的 KL 与 top-1 agreement；`inference_validation_before/after` 使用实际 dtype 的 norm/head，帮助识别 FP32 loss 与推理脱节。top-1 agreement 是所采样单步特征上的诊断，不是树接受率。
- **是否漂移**：weight version、optimizer steps、guard 拒绝次数、累计距离（启用相应 guard 时），以及沿 stream 的接受和耗时走势。训练 loss 下降不能单独证明泛化。
- **是否只是预算效应**：prompt/new tokens、EOS 与截断数量。reasoning 数据先用 1024 token 起跑，再对有希望的配置跑 2048 或更长预算；baseline 与候选必须同步改变预算，并给出截断比例。长 prompt 被容量拒绝会显式记录，不静默丢掉。

若 KL 降低但接受率没有变化，先看推理权重量化、局部留出一致率和深层 survival；后者下降时，下一步应研究 rollout-aware 目标，不能只继续增大学习率。若接受率提升但整体更慢，应优先降低更新频率、缩小 scope/训练样本，或单独报告冻结后缀收益；不能删除训练耗时后宣称在线加速。

`--profile` 对阶段同步计时，会扰动性能；只用于定位 target_prefill / target_verify / teacher_collection / feature_forward / head_update 等成本，正式吞吐另跑关闭 profile 的实验。

## 7. 已完成验证与尚未覆盖部分

2026-09-17：25 项 CPU 测试全部通过。覆盖标签配对、三层选择、压缩词表映射、分块/完整前缀特征一致性、零接受 root 监督、FP16 更新丢失复现、Adam 对照、guard/异常/replay 回滚、reset/persistent/freeze、EOS/容量限制、三个更新时间下与 target greedy 的一致性，以及真实本地小 checkpoint 的加载、评测、结果配对全流程。

这些测试不证明真实 8B 的收益、速度或显存足够，也不保证 FP16 GPU 在所有长序列上逐 token 一致。未启动真实大模型实验；需要在你的 GPU 和实际 checkpoint 上先跑 smoke。当前不保存适应后的权重和 Adam checkpoint，persistent 只在一次进程内有效；中断后应以新目录从同样初始权重完整重跑，不能直接跳到后半段续写结果。
