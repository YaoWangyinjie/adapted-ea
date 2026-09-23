"""Counter regression tests; no torch, weights, or GPU required."""
import ast
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("acceptance_metrics", ROOT / "eagle/evaluation/acceptance_metrics.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class AcceptanceMetricsTest(unittest.TestCase):
    def test_zero_based_and_explicit_counts(self):
        self.assertEqual(m.count_speculative_steps(0), 1)
        self.assertEqual(m.count_speculative_steps(-1), 0)
        self.assertEqual(m.count_speculative_steps(7, {"total_steps": 3}), 3)
        self.assertEqual(m.count_speculative_steps(7, idx_semantics="count"), 7)
        with self.assertRaises(ValueError):
            m.count_speculative_steps(-2)

    def test_root_truncation_and_aggregation(self):
        rows = [
            dict(qid=1, turn=0, total_accept_length=8, total_steps=2, new_tokens=9),
            dict(qid=2, turn=0, total_accept_length=3, total_steps=1, new_tokens=4),
        ]
        s = m.summarize_stats(rows)
        self.assertAlmostEqual(s["micro"]["accepted_children_per_step"], 11/3)
        self.assertAlmostEqual(s["micro"]["verified_tokens_per_step_including_root"], 14/3)
        self.assertAlmostEqual(s["micro"]["returned_tokens_per_step"], 13/3)
        self.assertEqual(s["discarded_verified_tokens"], 1)
        self.assertEqual(s["macro_question"]["returned_tokens_per_step"], 4.25)
        self.assertEqual(s["legacy_zero_based_denominator_demo"]["undefined_questions"], 1)
        with self.assertRaises(ValueError):
            m.summarize_stats([rows[0], rows[0]])
        with self.assertRaises(ValueError):
            m.summarize_stats([dict(total_accept_length=1, total_steps=1, new_tokens=3)])

    def test_sft_two_turns_and_single_step(self):
        rows = [
            dict(model_tag="baseline", new_tokens=10, idxs=[1, 1], avg_accept_len=5),
            dict(model_tag="baseline", new_tokens=4, idxs=[0], avg_accept_len=0),
        ]
        s = m.summarize_sft(rows, "zero-based")["baseline"]
        self.assertEqual(s["verification_steps"], 5)
        self.assertEqual(s["counter_tokens_per_step_micro"], 2.8)
        self.assertEqual(s["counter_tokens_per_step_macro"], 3.25)

    def test_ea4_summary_exposes_both_length_conventions(self):
        spec = importlib.util.spec_from_file_location("ea4_eval", ROOT / "eagle/evaluation/gen_ea_answer_ea4.py")
        evaluator = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(evaluator)
        row = dict(status="ok", total_accept_length=8, total_steps=2,
                   total_drafted_tokens=12, new_tokens=9, generation_seconds=1,
                   stop_reason="eos", adaptation={"status": "skipped"})
        out = evaluator.summarize([row])
        self.assertEqual(out["accepted_children_per_step"], 4)
        self.assertEqual(out["verified_tokens_per_step_including_root"], 5)
        self.assertEqual(out["returned_tokens_per_step"], 4.5)
        self.assertEqual(out["discarded_verified_tokens"], 1)
        self.assertEqual(out["tokens_per_second"], 9)

    def test_sft_forwards_token_cap_in_warmup_and_evaluation(self):
        source = ROOT / "eagle/sft_overfit/eval_sft.py"
        tree = ast.parse(source.read_text("utf8"))
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Attribute) and n.func.attr == "eagenerate"]
        self.assertEqual(len(calls), 2)
        for call in calls:
            kw = next(k for k in call.keywords if k.arg == "max_new_tokens")
            self.assertEqual(ast.unparse(kw.value), "args.max_new_tokens")


if __name__ == "__main__":
    unittest.main()
