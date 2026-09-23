# 验证记录

日期：2026-09-23。

独立项目已通过 31 项测试，记录在 `validation/pytest.xml`：

- 用固定的上游源码函数核对 Hedge softmax 和 ENS 累计 loss 权重；
- 对照原始递归层，验证七步前向及梯度；
- 核对 meta loss 的 epoch/batch/unroll 汇总；
- 核对浮点参数合并，以及 d2t/t2d 的类型和值；
- 微型真实模型完成两种 variant 的三块配对实验，包含生成、TraceDraft、三学习器训练、optimizer 续接、合并和统计；
- 两组生成序列及外层 checkpoint 完全一致；
- 验证中断后仅重跑未完成块；
- 核对三个学习器的全部计时，拒绝错误配对；
- 验证 TraceDraft 的 FP32 norm、小更新、诊断一致性、验证回退及 reservoir 抽样；
- 四卡队列按小数据集优先排序，并拒绝重复任务；
- 输出不一致时生成差异诊断，并隔离旧的成功报告。

两个 Bash 模型脚本通过 `bash -n` 检查。测试只将模型加载和硬件信息替换为微型 CPU 环境，实际使用生产 worker、生成、训练、合并和报告函数；无需下载模型。

本机真实 8B 功能验证另存于 `validation/real_8b/`，完成情况和实测结果以该目录中的 state、summary 和 comparison 为准。配置为 `configs/local_4060_smoke.json`；它使用现有 D 盘权重、NF4 GPU target、CPU draft，以及短回答，仅用于核验代码链路。

Vicuna-13B-v1.3 的 profile、模板和全词表逻辑有代码检查，尚未在本机下载该模型做真实权重实验。A800 上未量化、全 GPU 的正式性能需在服务器测量。

## 真实 8B 运行结果

两组的全部生成、三学习器训练和第二块合并均完成；外层权重完全一致。严格的完整输出比较未通过：第 83 题第一 turn 第 30 个 token 在 NF4/FP16 的不同验证路径中出现一个 FP16 刻度的 logit 差异，改变了 argmax。代码保留拒绝计算加速比的保护，并写入分歧诊断。详见 [本地验证摘要](validation/real_8b_summary.md)。


## 三组比较入口更新

新增 `method=tracedraft`、`run_comparison.py` 与 `scripts/run_comparison.sh`，以及组合方法相对独立 TraceDraft 的 `pairwise_gains`。本次完整回归为 **33 passed**（`validation/pytest.xml`）。新增测试对 hedge 和 ens 分别运行微型模型的完整三组流程，确认独立 TraceDraft 没有外层训练，外层对照的轨迹仍相同，并验证两种参考组的比值。

两个核心 Bash 脚本另以命令替身核对实际调用：OSD 仅两组及汇总，OnlineSPEC 调用三组入口；单卡编号、Vicuna profile、带空格路径、seed 与 resume 均正确转发。此次未重新运行真实 8B 性能实验，先前 NF4/FP16 完整输出分歧的记录仍然有效。
