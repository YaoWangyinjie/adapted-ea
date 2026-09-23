# 源码对应与实现约束

## OnlineSPEC 来源

仓库：https://github.com/ZinYY/OnlineSPEC  
检查版本：`e58f82eb3f3adca3a686211236bf4f6e9e7e3a2b`  
论文：https://arxiv.org/abs/2603.12617

| 上游源码 | 本项目 | 对应内容 |
|---|---|---|
| pipeline_eagle3_hedge.py: compute_softmax_weights_from_losses | ensemble.py: weights_for_block | 最近一块 loss 的 softmax |
| pipeline_eagle3_hedge.py: run_pipeline/train(use_meta_learner=True) | ensemble.py: training_parents；run.py | 三个学习器从本块合并 checkpoint 开始 |
| pipeline_eagle3_ens.py: compute_weights_from_cumulative_losses | ensemble.py: weights_for_block/next_history | 累计 loss 的指数权重 |
| pipeline_eagle3_ens.py: train | run.py + core/training.py | 各学习器独立续接自身 checkpoint 和 optimizer |
| traineagle3.py: ploss、total_loss | core/training.py；training.py: upstream_meta_loss | 带 0.8 衰减的训练目标；未加权的 meta loss |
| train_eagle3/cnets.py: LlamaDecoderLayeremb/forward | core/training.py: recurrent_step/sequence_loss | 七步递归注意力和反向传播 |
| pipeline_eagle3_*.py: ensemble_ea3_models | ensemble.py: merge_checkpoints | 按 meta 权重混合参数，输出一个服务 draft |

对应源码已复制到 `reference/EAGLE`，哈希在 `PROVENANCE.json`。不将 EAGLE-3 的七步训练误写为“七个模型的加权平均”：本实现有三个外层学习器，每个学习器的训练内部有七步递归预测。

## TraceDraft 接入点

`worker.py` 使用合并 checkpoint 创建服务模型，然后调用固定的 `core/worker.py::infer_block`。后者沿用 `core/backend/ea_model_4.py` 的 verified-prefix 采样和特征提取，交给 `core/backend/online_head.py` 更新 norm/head。

特征提取完成后恢复 tree mask；已保存的 KV 仅依赖冻结的 draft 主干，当前回答的 norm/head 更新不改变其含义。外层更新会改变 fc/midlayer，因此每个新块重新载入模型，并清空上一块的学习器和 replay。

当前回答内只更新一次。TraceDraft 训练的 prompt/response 位置预算、局部留出验证比例、更新时点和 optimizer 信息记录在每次更新的 config 中。

## 外层与内层隔离

服务进程不输出可供外层接着训练的 TraceDraft checkpoint。外层训练的父路径只可能是：

- Hedge：磁盘上的本块合并 checkpoint；
- ENS：上块对应 learner 的 checkpoint。

三份训练任务使用相同的生成数据、相同 shuffle seed，但学习率不同。合并 checkpoint 不继承任何一个 learner 的 Adam 状态；ENS 的三条 optimizer 状态则各自独立。

完整配对测试会核对每块合并权重、每个 meta loss 和 checkpoint 的实际文件 SHA256。这样，组合增益才来自服务阶段的 TraceDraft。

## 数值与实验协议

默认 `merge_precision=fp32`，离散词表映射必须一致。可选 `upstream_bf16` 复用上游浮点混合顺序；不会把整数映射转 BF16 再四舍五入。相同初始 draft 会作为三条学习路径的共同起点。

本项目固定生成 token/mask，避免同一份答案在两组中被不同的聊天重建规则解析。与官方预处理、数值精度、数据集或初始 checkpoint 有差别时，需要在实验设置里说明。

同样的参数传给两组。对比器拒绝配置、模型、数据、源码、完整输出序列或外层轨迹不一致的配对。现有配对默认采用 greedy；未实现把该验证机制自动推广到随机采样。

## 文件范围

新写的 ensemble、run、report、queue 模块位于 `onlinespec_trace/`。固定后端位于 `onlinespec_trace/core/`，来自本项目已测试的 OSD/TraceDraft 融合模块；只调整 Python import 的包路径，保留原方法标识和训练逻辑。原 `adapted-ea/osd_tracedraft` 与 `adapted-ea/eagle/model` 未修改。

## 独立 TraceDraft 对照与三组入口

`method=tracedraft` 将内层配置映射为固定后端的独立 TraceDraft，`has_outer=False`，整个运行不启动任何外层 learner 训练。它和其他组共用相同分块、初始服务权重的数值转换及推理后端；块内 Persistent 状态也在相同块边界重建。

`run_comparison.py` / `scripts/run_comparison.sh` 顺序运行 OnlineSPEC、独立 TraceDraft、OnlineSPEC+TraceDraft，统一配置后生成三组报告。报告以 OnlineSPEC 为参考，同时保存组合相对独立 TraceDraft 的 `pairwise_gains`。旧 `run_pair.py` 仍保留原两组语义。
