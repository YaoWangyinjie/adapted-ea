# 融合代码修复与验证

本次更新只作用于 `adapted-ea/osd_tracedraft`，未改动原 `eagle/model`、模型下载目录、论文或历史实验结果。

## 修改内容

1. **降低在线更新的附带开销。** Prompt 先执行原有 reservoir 抽样，再对保留位置计算 teacher logits；验证阶段已有的 logits 只复制抽中位置。保持原采样 RNG 与最终槽位一致。默认 basic 诊断保留实际推理验证和更新检查，完整诊断可按频率开启。
2. **让推理验证参与更新决策。** TraceDraft 在本次独立留出位置上比较候选更新前后的实际 norm/head KL，包含推理精度转换。未通过时恢复 live/master 权重、Adam 矩、步数、版本、回放及 RNG。没有足够验证位置时跳过。
3. **保留 norm 的小幅更新。** 最终输出 norm 保持 FP32，直接读取 checkpoint 的 norm 权重；归一化激活仍转回 head 的推理 dtype。其它 draft 层与 head 不因此转为 FP32。所有对照组统一该设置。
4. **保护块间 OSD 更新。** 默认学习率改为 1e-5。按完整 conversation 留出验证题目，比较相同位置上的 FP32 七步加权 CE；更新退化、非有限或验证集不可用时，下一块使用父 checkpoint 及原 optimizer。日志包含验证题目、前后 loss、阈值、实际提交与尝试步数以及拒绝原因。
5. **明确日志口径。** basic 下大 head 的变化比例是抽样估计，字段为 `inference_changed_fraction_sampled`，精确值留 null；完整诊断提供精确比例。失败更新与验证成本仍计入时间。

## 融合结构

```text
当前块 checkpoint
  ├─ 推理：冻结 target，EAGLE-3 draft 提议 → target 验证
  │    └─ TraceDraft：当前已确认前缀 → norm/head 候选更新 → 局部验证 → 提交/回退
  └─ 块结束：另起训练进程，重新载入该块初始 checkpoint
       └─ 生成记录分为训练题和验证题 → OSD 七步蒸馏 → 块级验证 → 新 checkpoint/父 checkpoint
            └─ 下一块重新加载，重新建立 TraceDraft 学习器
```

TraceDraft 不训练 fc/midlayer，当前回答的 KV cache 可继续使用。OSD 更新 fc/midlayer 后重新加载模型并清空旧 replay，避免把旧特征交给新主干。Reset 在新 conversation 恢复当前 OSD checkpoint；`block_persistent` 只在同一块内延续。验证保护是本实现新增的训练策略，实验命名应注明 guard，不能直接称为原论文 OSD 基线的原样复现。

## 运行与记录

服务器运行方式保持原入口与脚本，新增参数和关闭选项见上级 README。更换源码或参数请使用新输出目录。

本目录的 `osd.json`、`osd_tracedraft.json` 是本机 4060 的小规模配置：DeepSeek-R1-Distill-Llama-8B target 使用 NF4 并放 GPU，匹配 draft 放 CPU、4 线程；MT-Bench 3 个 conversation、每个 2 turn，每 turn 最多 64 tokens，block size=2。

在 `adapted-ea` 目录依次运行：

```powershell
$env:CUDA_VISIBLE_DEVICES="0"
$env:HF_HUB_OFFLINE="1"
$env:TRANSFORMERS_OFFLINE="1"
python -m osd_tracedraft.run --config osd_tracedraft/local_smoke_4060/fix_validation/osd.json
python -m osd_tracedraft.run --config osd_tracedraft/local_smoke_4060/fix_validation/osd_tracedraft.json
python -m osd_tracedraft.report --reference-method osd --runs osd_tracedraft/local_smoke_4060/fix_validation/osd osd_tracedraft/local_smoke_4060/fix_validation/osd_tracedraft --output osd_tracedraft/local_smoke_4060/fix_validation/comparison
```

已有运行结果不覆盖；复现实验需给配置中的 output 换新目录。两组各自保留 manifest、输入输出 token IDs、训练记录、验证日志、checkpoint 及配对统计。`source_before/` 是修复前 Python 源码快照，`overhead/report.json` 是独立单题诊断，`pytest.xml` 是自动化测试记录。

## 单题诊断结果

使用同一初始 checkpoint、同一 MT-Bench 首题第一 turn，分别运行关闭更新、完整诊断和精简诊断。三次均生成相同的 64 个 token。

| 模式 | 生成总耗时（秒） | feature forward（秒） | head update（秒） |
|---|---:|---:|---:|
| baseline | 9.3193 | 0.0000 | 0.0000 |
| full | 15.8113 | 0.6408 | 4.6331 |
| basic | 13.6620 | 0.5428 | 3.4142 |

本次精简诊断的 head update 用时比完整诊断降低 26.3%。这是一次功能诊断的计时，不是重复性能测量；加入训练后总耗时仍高于关闭更新。精简与完整模式的 head SHA256 完全一致，norm 实际变化范数均为 0.00063551。两种模式的实际推理验证 KL 都从 0.536660 降至 0.534691。

检查同时确认：冻结主干 SHA256 不变；target 参数不参与训练；特征提取前后的 KV cache 对象和值不变；tree mask 恢复。原始数据在 `overhead/report.json`。

## 完整配对验证结果

两组均完成 3 个 conversation、6 个 turn、384 个生成 token；每 turn 前 16 个 token 的 target-only greedy 检查通过，两组全部生成 token 的 SHA256 相同。

| 组别 | 平均长度（children） | 生成 tok/s | 相对 OSD | 含训练 tok/s | Trace 提交/回退 | OSD 提交/尝试步数 |
|---|---:|---:|---:|---:|---:|---:|
| osd | 4.7714 | 5.5154 | 1.0000× | 3.0111 | 0/0 | 0/1 |
| osd_tracedraft | 4.7714 | 3.8734 | 0.7023× | 2.4308 | 3/3 | 0/1 |

两组相同的 OSD 候选更新，使整题留出 CE 从 0.845005 变为 0.850773，因此均回退。两组下一块 checkpoint 与原始 draft 权重 SHA256 相同，回退后正常生成。此处 OSD 最终提交步数为 0，不能把该运行写成‘成功完成 OSD 学习后的增益’。

该检查用于验证融合逻辑、更新精度、回退和记录链路。当前短输出和 CPU draft 条件下，TraceDraft 仍有额外训练成本；上述数字不代表 A800 全 GPU 部署的吞吐。冻结骨干、KV 保留和诊断一致性的原始结果见 `verification.json` 与 `overhead/report.json`；完整统计见 `comparison/`。

自动化测试共 42 项通过，覆盖更新数值、推理精度、事务回滚、完整 conversation 留出、checkpoint/optimizer 恢复、七步递归对齐、输出计数及原有运行流程；JUnit 结果见 `pytest.xml`。Vicuna 真实权重和全 GPU/A800 路径仍需在对应服务器执行 smoke。
