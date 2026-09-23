"""Summarize a completed functional smoke and its strict comparison outcome."""
from pathlib import Path
from onlinespec_trace.core.io_utils import read_json,read_jsonl

ROOT=Path(__file__).resolve().parent

def main():
    root=ROOT/"validation/real_8b"
    failure=read_json(root/"comparison/comparison_failure.json")
    outer=read_json(root/"outer_checkpoint_audit.json")
    diagnostic=read_json(ROOT/"validation/numerical_diagnostic/report.json")
    assert failure["status"]=="invalid_pair" and not failure["speedup_reported"]
    assert all(r["identical"] for r in outer)
    lines=["# 本地真实 8B 功能验证","",
        "设备为 RTX 4060 Laptop 8GB；target 使用 NF4、FP16 计算，draft 放 CPU，4 个线程。两组各完成 MT-Bench 3 个 conversation、6 个 turn，每 turn 最多 32 tokens。",
        "外层使用 hedge，block size=2，三个学习率为 1e-5、3e-6、1e-6，各训练 1 epoch。TraceDraft 使用 Reset、lr=1e-5，短测中在第 2 次验证后更新；正式配置默认是第 5 次。","",
        "**生成、三学习器训练、模型合并和下一块续接均完成；严格的完整输出比较没有通过，因此不报告本次组合加速比。**","",
        "| 方法 | 生成 tokens | Trace 提交/回退 | 外层 optimizer steps |",
        "|---|---:|---:|---:|"]
    for method in ("onlinespec","onlinespec_trace"):
        summary=read_json(root/method/"summary.json")
        assert summary["status"]=="complete"
        lines.append(f"| {method} | {summary['new_tokens']} | {summary['trace_updated_turns']}/{summary['trace_rolled_back_turns']} | {summary['outer_optimizer_steps']} |")
    lines += ["","## 完整输出分歧","",
        "第 83 题第一 turn 的第 30 个 token 出现 but / and 分歧，随后第 32 个 token 也不同。每 turn 前 16 个 token 的检查均通过，但不足以发现这处差异。",
        "独立诊断重新运行了这道题，并验证了分歧之前的完整前缀相同。观察到的 target logits 如下：","",
        "| 执行路径 | but logit | and logit | argmax |","|---|---:|---:|---|"]
    c=diagnostic["canonical_shared_prefix"]
    lines.append(f"| 对共同前缀单独计算 target | {c['but_logit']:.6f} | {c['and_logit']:.6f} | but |")
    for r in diagnostic["records"]:
        obs=r["observations"][0];assert obs["matches_shared_prefix"]
        lines.append(f"| {r['method']} 的验证树 | {obs['but_logit']:.6f} | {obs['and_logit']:.6f} | {'but' if obs['argmax']==719 else 'and'} |")
    lines += ["",
        "差异是一个 FP16 刻度（0.015625）：组合组路径中两个候选的 logits 被舍入成相同值，argmax 选择了 token ID 较小的 and。这定位了本次 NF4/FP16 执行中的数值分歧；没有据此放宽完整输出检查。",
        "两个合并 checkpoint 和三个学习器 checkpoint 的实际 SHA256 全部一致，三个 meta loss 与合并权重也一致。TraceDraft 临时权重没有写回外层。",
        "原始结果保留在 validation/real_8b/。comparison_failure.json 记录具体分歧位置；没有为这次不匹配的输出生成加速比。validation/numerical_diagnostic/report.json 保留独立诊断。",
        "正式性能实验应在服务器上执行并检查完整输出；短测建议 verify_tokens 与 max_new_tokens 相同。本次记录证明了融合链路可执行，也记录了量化环境的正确性边界，不能作为 TraceDraft 正向提升的证据。"]
    (ROOT/"validation/real_8b_summary.md").write_text("\n".join(lines)+"\n",encoding="utf8")
    print("Saved functional validation and numerical mismatch without a speedup claim.")

if __name__=="__main__":main()
