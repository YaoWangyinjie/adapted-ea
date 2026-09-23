# RTX 4060 本地功能检查

模型权重存放在 D:/model_weights。两个模型目录均包含 download_provenance.json，记录下载版本、文件大小和 SHA-256。原始权重保持不变，NF4 量化在加载时进行。

## 本次配置

- Target：DeepSeek-R1-Distill-Llama-8B，加载为 NF4，放在 CUDA:0。
- Draft：官方匹配的 EAGLE-3，放在 CPU；OSD 训练使用 FP32。
- MT-Bench：question_id 81、82、83，各保留两个 turn；每个 turn 最多生成 64 个新 token，上下文上限 512。
- 树：60 个节点、深度 5、top-k 10。
- OSD：每块 2 个 conversation，lr=1e-3，1 epoch，batch=2，七步展开，衰减 0.8。
- TraceDraft：Reset，lr=1e-5，在第 5 次验证后更新一次，prompt/response 各最多 64 个位置；R/A 关闭。
- 两组均对每个 turn 的前 16 个输出 token 进行 target greedy 一致性检查。
- 这是短输出功能测试，不能作为 A800 正式实验的性能数据。

## 两个更新过程怎样衔接

OSD 先完成一个请求块的生成，保存实际 token 序列和当前回答的训练掩码。训练进程重新载入该块初始 checkpoint，用冻结 target 的分布监督 draft，训练 fc、midlayer、norm 和 lm_head。七步表示递归预测展开，损失按 0.8^r 加权，不是七层模型平均。最后一块没有后续问题，因此不再训练。

组合组在上述推理阶段增加 TraceDraft。它在当前回答内部使用 prompt 和已经验证的前缀，更新 draft 的 norm/head。新 conversation 恢复到本块初始 checkpoint，同一 conversation 的第二个 turn 继续使用临时更新。OSD 的训练起点仍然是块初始 checkpoint；这避免临时更新改变两组 OSD 对照的训练起点。

本次共 3 个 conversation：先生成前两个，完成一次 OSD 块训练，再用更新后的 checkpoint 生成第三个。这使实际训练、checkpoint 保存与重载都被执行。训练日志记录优化步骤、损失、梯度范数和监督位置数，TraceDraft 日志记录更新/拒绝及参数变化。

## 与 OnlineSPEC 的关系

OnlineSPEC 论文的在线学习框架允许在投机验证轮之间更新 draft，因此不能把“回答内更新”描述为整个 OnlineSPEC 理论框架完全没有的能力。它公开的 EAGLE-3 实验管线按请求块生成数据、训练后续块，这一具体实现可以与本项目的回答内适应结合。

本目录实现的是参考 OnlineSPEC pipeline_eagle3.py 的单 draft OSD-EAGLE-3 适配，不包含其 Ens-EAGLE-3 多学习器算法。

若进一步结合 Ens-EAGLE-3，可在块开始时合并慢速学习器，复制出一个服务 draft；TraceDraft 只更新该服务副本的 norm/head。块结束后，各慢速学习器用真实生成数据训练并更新组合权重。下一块重新合并并建立服务副本。主干改变后，应清空基于旧主干的特征回放、重建快更新优化器和 anchor。这样可以分别研究块间学习与回答内适应的贡献，但不能直接宣称原文理论保证覆盖新增更新。

组合提升可支持“方法互补”的结论。完整比较应包含同配置的 EAGLE-3、TraceDraft、OnlineSPEC 和 OnlineSPEC+TraceDraft，并计入训练时间。仅比较 OSD 与组合组时，得到的是在该 OSD 适配基础上的增量效果。

来源：[OnlineSPEC 论文](https://arxiv.org/html/2603.12617v1)、[OnlineSPEC 源码](https://github.com/ZinYY/OnlineSPEC)、[原始 OSD 源码](https://github.com/LiuXiaoxuanPKU/OSD)。

## 执行与统计

在 E:/实验室/Adapted_ea/adapted-ea 目录中，用 PowerShell 执行：

~~~powershell
$env:CUDA_VISIBLE_DEVICES='0'
$env:HF_HOME='D:\model_weights\.cache\huggingface'
$env:HF_HUB_OFFLINE='1'
$env:TRANSFORMERS_OFFLINE='1'
$py='E:\实验室\Adapted_ea\tmp\osd_tracedraft_test_env\Scripts\python.exe'
& $py -m osd_tracedraft.run --config osd_tracedraft/local_smoke_4060/osd.json
& $py -m osd_tracedraft.run --config osd_tracedraft/local_smoke_4060/osd_tracedraft.json
& $py -m osd_tracedraft.report --runs osd_tracedraft/local_smoke_4060/osd osd_tracedraft/local_smoke_4060/osd_tracedraft --reference-method osd --output osd_tracedraft/local_smoke_4060/comparison
~~~

已存在的完整实验目录不会被覆盖。相同代码和配置可使用 --resume；修改代码后应使用新的输出目录。

每组结果包含 config、manifest、question_order、blocks、summary。blocks 下保存 answers、turns、trace_updates、training_records、OSD 更新日志、硬件信息及训练 checkpoint。

平均接受长度为 accepted children/verification steps；含 root 的长度另列。吞吐分别统计生成过程、计入 OSD 训练与会话 reset、整个执行管线。组合组的加速比以 OSD 组为 1.000；不是相对于普通自回归生成的加速比。统计器要求两组生成 token 完全一致，否则停止计算配对加速比。

## 已完成的代码验证

本次环境使用 Python 3.12、PyTorch 2.6.0+cu124、Transformers 4.53.1、bitsandbytes 0.48.2、Accelerate 1.10.1。22 项测试通过，包含真实 CUDA 的 target/CPU draft 设备传递与反向传播、小模型 NF4 加载，以及 OSD 参考组的两组统计检查。真实 8B 权重测试已完成，详见 [RESULTS.md](RESULTS.md)。
