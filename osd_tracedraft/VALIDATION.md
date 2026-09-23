# 验证记录

历史验证日期：2026-09-22。

2026-09-23 更新：已补充真实 DeepSeek 8B 权重的 RTX 4060 Laptop 功能验证，以及更新保护和精度修复的 42 项测试；新记录见 [修复验证报告](local_smoke_4060/fix_validation/README.md)。下文保留最初的微型模型测试记录。

本机使用 Python 3.12、PyTorch 2.6.0+cpu、Transformers 4.53.1。测试使用随机初始化的微型 Llama target 和 EAGLE-3 draft，不下载正式权重。

## 已验证

- 20 项测试通过，原始输出与 JUnit 报告在 validation/ 下。
- 真实小模型完成 baseline、TraceDraft、OSD、OSD+TraceDraft 四组的生成、分块训练、checkpoint 保存、续接和配对统计。
- OSD 与组合组在相同生成数据下得到相同的外层 checkpoint，确认临时 head 更新不污染 OSD 起点。
- 对照本地 OnlineSPEC 原始源码：七次递归前向与梯度一致。
- 使用与推理相同的 draft 参数：递归第一步与原始 draft 前向一致。
- 开启与关闭梯度 checkpoint 的七步梯度一致。
- token、teacher hidden、下一位置监督、assistant mask 对齐。
- 使用生产代码同款 from_pretrained 接口保存并重新加载 target/draft，生成 token 完全一致。
- 原始 target 与 token embedding 在 OSD 中保持冻结。
- OSD AdamW 状态跨块续接；训练从 FP32 checkpoint 读取，不经过推理精度量化。
- Reset 在 conversation 边界生效，同一个 MT-Bench conversation 的第二 turn 保留第一个 turn 的更新。
- 块内 Persistent 的 replay 抽样、字节上限与 anchor loss 生效并记录。
- Llama3 RoPE 适配器与 Transformers 实现一致。
- 原始计数、root、截断扣减及训练开销的汇总关系通过验证；矛盾计数被拒绝。
- 断点续跑只重做未完成块；四卡队列为每个进程设置单一可见 GPU，失败任务后继续调度。
- 三份 Bash 脚本均通过 bash -n，全部 Python 文件语法检查通过。
- 两个模型 profile 的 MT-Bench dry run 均为 80 个 conversation、160 个 turn、2 个推理块、1 个 OSD 训练块。
- 独立复制整个目录后，19 项测试通过、1 项跳过；跳过的是需要外部 OnlineSPEC 源码的对照测试，其余测试不依赖外部 eagle/。

## 服务器上的最后一步

最初测试未运行正式模型权重；此后已完成 DeepSeek-R1-Distill-Llama-8B 的本地 NF4 target / CPU draft 测试。Vicuna-13B-v1.3 真实权重和全 GPU/A800 性能仍未验证。请按 README 先跑 4 题、block_size=2 的 smoke 实验，核对配套 EAGLE-3 draft、greedy 一致性与 OSD 保存/重新加载，然后再运行完整队列。

CPU 测试时间不代表真实吞吐，不作为论文实验结果。validation/mtbench_dry_run.json 只是执行计划；正式数据写入你指定的输出目录。
