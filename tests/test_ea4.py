"""CPU tests with real PyTorch and randomly initialized local EAGLE/Llama models.

Run: python -m unittest discover -s tests -p test_ea4.py -v
No downloaded model weights, datasets, GPUs, or network are used.
"""
import copy
import json
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn

from eagle.model.online_head import AdaptationConfig, OnlineHead, PositionReservoir, adam_candidate
from eagle.model.ea_model_4 import EaModel, select_eagle_hidden
from eagle.evaluation.gen_ea_answer_ea4 import summarize
from eagle.evaluation.plan_ea4_experiments import cases, main as plan_main
from eagle.evaluation.gen_ea_answer_ea4 import parse_args, load_questions
from eagle.evaluation.gen_ea_answer_ea4 import main as evaluate_main
from eagle.evaluation.compare_ea4 import compare

torch.set_num_threads(1)


class Norm(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.variance_epsilon = 1e-5

    def forward(self, x):
        normalized = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.variance_epsilon)
        return self.weight * normalized.to(x.dtype)


def learner(precision='master_fp32', **kwargs):
    torch.manual_seed(7)
    norm = Norm(4)
    head = nn.Linear(4, 6, bias=False)
    cfg = AdaptationConfig(precision=precision, **kwargs)
    return OnlineHead(norm, head, cfg)


def examples(n=16):
    generator = torch.Generator().manual_seed(42)
    teacher = torch.randn(6, 4, generator=generator)
    features = torch.randn(n, 4, generator=generator)
    return [{'feature': x.clone(), 'target': teacher @ x, 'group': 'response', 'valid': True, 'position': i + 1}
            for i, x in enumerate(features)]


class EngineTests(unittest.TestCase):
    def test_hidden_selection_does_not_take_final_norm(self):
        states = [torch.full((1, 3, 2), float(i)) for i in range(4)]
        result = select_eagle_hidden(states, 'cpu', 6)
        self.assertTrue(torch.equal(result[0, 0], torch.tensor([0., 0., 1., 1., 2., 2.])))
        with self.assertRaises(ValueError):
            select_eagle_hidden(states[:2], 'cpu', 4)

    def test_label_alignment_and_response_boundary(self):
        cfg = AdaptationConfig(source='balanced', validation_fraction=0)
        collector = PositionReservoir(cfg, torch.ones(6, dtype=torch.bool), 3, 1)
        targets = torch.arange(30).reshape(5, 6).float()
        collector.observe(targets, 0)
        rows = collector.selected(5)
        self.assertEqual([r['position'] for r in rows], [1, 2, 3, 4])
        self.assertEqual([r['feature_index'] for r in rows], [0, 1, 2, 3])
        self.assertEqual([r['group'] for r in rows], ['prompt', 'response', 'response', 'response'])
        for row in rows:
            self.assertTrue(torch.equal(row['target'], targets[row['position']]))
        collector.config.alignment = 'legacy'
        self.assertEqual([r['feature_index'] for r in collector.selected(5)], [1, 2, 3])

    def test_reservoir_bounded_repeatable_and_rng_isolated(self):
        cfg = AdaptationConfig(source='balanced', max_prompt_positions=3, max_response_positions=4)
        before = random.getstate()
        def collect():
            obj = PositionReservoir(cfg, torch.ones(6, dtype=torch.bool), 20, 1)
            obj.observe(torch.zeros(100, 6), 0)
            return obj
        a, b = collect(), collect()
        self.assertEqual(random.getstate(), before)
        self.assertEqual(len(a.rows['prompt']), 3)
        self.assertEqual(len(a.rows['response']), 4)
        self.assertEqual([r['position'] for r in a.selected(100)], [r['position'] for r in b.selected(100)])

    def test_functional_adam_matches_torch_and_does_not_mutate_inputs(self):
        torch.manual_seed(3)
        parameter = nn.Parameter(torch.randn(4, 6))
        initial = parameter.detach().clone()
        optimizer = torch.optim.Adam([parameter], lr=3e-4)
        master, state = {'w': initial.clone()}, {}
        for step in range(1, 9):
            gradient = torch.randn_like(parameter)
            before = master['w'].clone()
            candidate, state_new = adam_candidate(master, {'w': gradient}, state, step, 3e-4)
            self.assertTrue(torch.equal(master['w'], before))
            parameter.grad = gradient.clone()
            optimizer.step()
            torch.testing.assert_close(candidate['w'], parameter.detach(), atol=2e-7, rtol=1e-6)
            master, state = candidate, state_new

    def test_learning_reduces_fixed_teacher_kl_and_holds_out_positions(self):
        obj = learner(learning_rate=.02, max_step_drift=0, train_steps=5, replay_bytes=2000)
        report = obj.update(examples(20))
        self.assertEqual(report['status'], 'updated')
        self.assertLess(report['train_after']['kl'], report['train_before']['kl'])
        self.assertEqual(report['validation_positions'], 4)
        self.assertLessEqual(obj.history_bytes, 2000)
        self.assertEqual(len(obj.history), report['train_positions'])

    def test_fp32_master_preserves_updates_that_fp16_roundtrip_loses(self):
        def make(mode):
            norm = Norm(2).half()
            head = nn.Linear(2, 2, bias=False).half()
            with torch.no_grad():
                head.weight.copy_(torch.tensor([[1., 0.], [0., 1.]]))
            return OnlineHead(norm, head, AdaptationConfig(scope='norm_only', precision=mode,
                learning_rate=1e-6, validation_fraction=0, max_step_drift=0, gradient_clip=0))
        master, rounded = make('master_fp32'), make('roundtrip')
        rows = [{'feature': torch.tensor([1., 2.]), 'target': torch.tensor([8., -8.]),
                 'group': 'response', 'valid': True}]
        for _ in range(300):
            master.update(rows)
            rounded.update(rows)
        self.assertFalse(torch.equal(master.master['norm'], torch.ones(2)))
        self.assertTrue(torch.equal(rounded.master['norm'], torch.ones(2)))
        self.assertFalse(torch.equal(master.norm.weight, torch.ones(2).half()))

    def test_guard_rolls_back_weights_moments_and_version(self):
        obj = learner(learning_rate=.01, max_step_drift=0, validation_fraction=0)
        self.assertEqual(obj.update(examples())['status'], 'updated')
        old = {k: v.detach().clone() for k, v in obj.master.items()}
        moments = copy.deepcopy(obj.moments)
        step, version = obj.step, obj.version
        obj.config.max_step_drift = 1e-20
        report = obj.update(examples())
        self.assertEqual(report['status'], 'rolled_back')
        self.assertEqual((obj.step, obj.version), (step, version))
        for k, value in old.items():
            self.assertTrue(torch.equal(obj.master[k], value))
            self.assertTrue(torch.equal(obj.live[k], value.to(obj.live[k].dtype)))
        for k in moments:
            for before, after in zip(moments[k], obj.moments[k]):
                self.assertTrue(torch.equal(before, after))

    def test_runtime_failure_after_publish_is_transactional(self):
        obj = learner(learning_rate=.01, max_step_drift=0, validation_fraction=0)
        original = {k: v.detach().clone() for k, v in obj.master.items()}
        publish = obj._publish
        attempts = []
        def fail_once(parameters):
            publish(parameters)
            attempts.append(1)
            if len(attempts) == 1:
                raise RuntimeError('injected publication failure')
        with patch.object(obj, '_publish', side_effect=fail_once):
            with self.assertRaises(RuntimeError):
                obj.update(examples())
        self.assertEqual(obj.step, 0)
        self.assertEqual(obj.version, 0)
        self.assertEqual(obj.moments, {})
        for k, value in original.items():
            self.assertTrue(torch.equal(obj.master[k], value))
            self.assertTrue(torch.equal(obj.live[k], value))

    def test_reset_clears_optimizer_and_replay(self):
        obj = learner(learning_rate=.01, max_step_drift=0, replay_bytes=10000)
        obj.update(examples())
        self.assertGreater(len(obj.moments), 0)
        self.assertGreater(len(obj.history), 0)
        obj.reset()
        self.assertEqual(obj.moments, {})
        self.assertEqual(obj.history_bytes, 0)
        self.assertEqual(obj.version, 0)
        for k, value in obj.initial.items():
            self.assertTrue(torch.equal(obj.live[k], value.to(obj.live[k].dtype)))

    def test_history_failure_restores_replay_and_rng(self):
        obj = learner(learning_rate=.01, max_step_drift=0, replay_bytes=2000)
        obj.update(examples())
        old_history, old_bytes = list(obj.history), obj.history_bytes
        old_rng, old_version = obj.rng.getstate(), obj.version
        add = obj._add_history
        def fail_after_add(rows):
            add(rows)
            raise RuntimeError('injected history failure')
        with patch.object(obj, '_add_history', side_effect=fail_after_add):
            with self.assertRaises(RuntimeError):
                obj.update(examples())
        self.assertEqual(obj.version, old_version)
        self.assertEqual(obj.rng.getstate(), old_rng)
        self.assertEqual(obj.history_bytes, old_bytes)
        self.assertEqual([id(r) for r, _ in obj.history], [id(r) for r, _ in old_history])

    def test_balanced_loss_is_not_dominated_by_prompt_length(self):
        obj = learner(source='balanced', prompt_weight=.25)
        rows = examples(4)
        rows[0]['group'] = 'prompt'
        original = obj.evaluate(rows)['kl']
        duplicated = obj.evaluate(rows + [rows[0]] * 20)['kl']
        self.assertAlmostEqual(original, duplicated, places=5)

    def test_invalid_positions_do_not_update(self):
        obj = learner()
        rows = examples(2)
        rows[0]['valid'] = False
        rows[1]['feature'][0] = float('nan')
        self.assertEqual(obj.update(rows)['status'], 'skipped')
        self.assertEqual(obj.version, 0)


def tiny_model():
    from transformers import LlamaConfig
    from eagle.model.modeling_llama_kv import LlamaForCausalLM
    from eagle.model.configs import EConfig
    from eagle.model.cnets import Model
    torch.manual_seed(123)
    config = LlamaConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
        num_hidden_layers=8, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=128, attention_dropout=0.0)
    config._attn_implementation = 'eager'
    target = LlamaForCausalLM(config).float().eval()
    draft_cfg = EConfig(vocab_size=32, draft_vocab_size=32, hidden_size=16,
        intermediate_size=32, num_hidden_layers=1, num_attention_heads=4,
        num_key_value_heads=2, max_position_embeddings=128)
    draft = Model(draft_cfg, load_emb=False, total_tokens=5, depth=2, top_k=2).float().eval()
    draft.t2d.fill_(True)
    tokenizer = SimpleNamespace(eos_token_id=None, get_vocab=lambda: {})
    return EaModel(target, draft, tokenizer)


class DecoderTests(unittest.TestCase):
    def setUp(self):
        self.model = tiny_model()
        self.prompt = torch.tensor([[1, 4, 7, 9]])

    def test_no_update_matches_target_and_has_no_training_work(self):
        for n in (1, 7, 15):
            reference = self.model.naivegenerate(self.prompt, n, 40)
            output, generated, _, stats = self.model.eagenerate(self.prompt, max_new_tokens=n, max_length=40, log=True, enable_adaptation=False)
            self.assertTrue(torch.equal(reference, output))
            self.assertEqual(generated, n)
            self.assertEqual(stats['stage_calls']['target_prefill'], 1)
            self.assertNotIn('teacher_collection', stats['stage_calls'])
            self.assertNotIn('feature_forward', stats['stage_calls'])
            self.assertIsNone(self.model.learner)

    def test_all_update_schedules_preserve_greedy_output_and_target(self):
        target_before = {k: v.clone() for k, v in self.model.base_model.state_dict().items()}
        reference = self.model.naivegenerate(self.prompt, 12, 40)
        for schedule in ('prefill', 'warmup', 'end'):
            self.model = tiny_model()
            cfg = AdaptationConfig(update_at=schedule, source='balanced', warmup_steps=1,
                validation_fraction=0, learning_rate=.001, max_step_drift=0, feature_chunk_size=2)
            self.model.setup_online_adaptation(cfg)
            output, _, _, stats = self.model.eagenerate(self.prompt, max_new_tokens=12, max_length=40, log=True, profile=True)
            self.assertTrue(torch.equal(reference, output), schedule)
            self.assertEqual(stats['adaptation']['status'], 'updated', schedule)
            self.assertEqual(stats['stage_calls']['target_prefill'], 1)
            self.assertEqual(stats['adaptation_state']['weight_version'], 1)
            for name, tensor in self.model.base_model.state_dict().items():
                self.assertTrue(torch.equal(tensor, target_before[name]))

    def test_zero_accepted_children_still_supply_response_training(self):
        self.model.setup_online_adaptation(AdaptationConfig(validation_fraction=0, max_step_drift=0))
        def force_reject(logits, candidates, processor):
            return torch.tensor(0), torch.tensor(0), logits[0, 0]
        with patch('eagle.model.utils.evaluate_posterior', side_effect=force_reject):
            output, _, _, stats = self.model.eagenerate(self.prompt, max_new_tokens=8, max_length=40, log=True)
        self.assertEqual(stats['total_accept_length'], 0)
        self.assertGreater(stats['adaptation']['valid_positions'], 0)
        self.assertEqual(stats['adaptation']['status'], 'updated')
        self.assertTrue(torch.equal(output, self.model.naivegenerate(self.prompt, 8, 40)))

    def test_persistent_reset_and_freeze(self):
        self.model.setup_online_adaptation(AdaptationConfig(validation_fraction=0, max_step_drift=0))
        for expected in (1, 2):
            self.model.eagenerate(self.prompt, max_new_tokens=6, max_length=40)
            self.assertEqual(self.model.learner.version, expected)
        old = self.model.learner.master['weight'].detach().clone()
        self.model.eagenerate(self.prompt, max_new_tokens=6, max_length=40, enable_adaptation=False)
        self.assertTrue(torch.equal(self.model.learner.master['weight'], old))
        self.model.reset_online_adaptation()
        self.assertEqual(self.model.learner.version, 0)

    def test_context_and_eos_are_exact(self):
        out, n, _, stats = self.model.eagenerate(self.prompt, max_new_tokens=20, max_length=7, log=True)
        self.assertEqual(n, 3)
        self.assertEqual(stats['stop_reason'], 'context_limit')
        self.assertTrue(torch.equal(out, self.model.naivegenerate(self.prompt, 20, 7)))
        first = int(self.model.naivegenerate(self.prompt, 1, 40)[0, -1])
        self.model.tokenizer.eos_token_id = first
        out, n, _, stats = self.model.eagenerate(self.prompt, max_new_tokens=20, max_length=40, log=True)
        self.assertEqual(n, 1)
        self.assertEqual(stats['stop_reason'], 'eos')

    def test_contiguous_features_equal_individual_prefix_features(self):
        with torch.no_grad():
            _, hidden = self.model._target(self.prompt)
            full = self.model.ea_layer(hidden[:, :-1], input_ids=self.prompt[:, 1:], use_cache=False)
            for position in range(1, self.prompt.shape[1]):
                prefix = self.model.ea_layer(hidden[:, :position], input_ids=self.prompt[:, 1:position + 1], use_cache=False)
                torch.testing.assert_close(full[:, position - 1], prefix[:, -1], atol=1e-6, rtol=1e-5)
            collector = PositionReservoir(AdaptationConfig(source='balanced', feature_chunk_size=2),
                                           self.model.vocab_mask, 4, 1)
            collector.observe(torch.zeros(4, 32), 0)
            marker = object()
            self.model.ea_layer.stable_kv = marker
            rows = self.model._features(self.prompt, hidden, collector)
            self.assertIs(self.model.ea_layer.stable_kv, marker)
            for row in rows:
                torch.testing.assert_close(row['feature'], full[0, row['feature_index']], atol=1e-6, rtol=1e-5)

    def test_compact_draft_vocab_preserves_target_output(self):
        draft = self.model.ea_layer
        draft.config.draft_vocab_size = 16
        draft.lm_head = nn.Linear(16, 16, bias=False)
        draft.t2d = torch.arange(32) % 2 == 0
        draft.d2t = torch.arange(16)  # reduced index i maps to target index 2*i
        model = EaModel(self.model.base_model, draft, self.model.tokenizer)
        model.setup_online_adaptation(AdaptationConfig(validation_fraction=0, max_step_drift=0))
        reference = model.naivegenerate(self.prompt, 16, 40)
        out, _, _, stats = model.eagenerate(self.prompt, max_new_tokens=16, max_length=40, log=True)
        self.assertTrue(torch.equal(reference, out))
        self.assertGreater(stats['adaptation']['mean_draft_vocab_mass'], 0)
        self.assertLess(stats['adaptation']['mean_draft_vocab_mass'], 1)

    def test_local_bin_checkpoint_loader(self):
        model = self.model
        with tempfile.TemporaryDirectory() as directory:
            model.ea_layer.config.save_pretrained(directory)
            state = model.ea_layer.state_dict()
            del state['embed_tokens.weight']
            torch.save(state, Path(directory) / 'pytorch_model.bin')
            with patch('transformers.AutoConfig.from_pretrained', return_value=model.config), \
                 patch('transformers.AutoTokenizer.from_pretrained', return_value=model.tokenizer), \
                 patch('eagle.model.modeling_llama_kv.LlamaForCausalLM.from_pretrained', return_value=model.base_model):
                restored = EaModel.from_pretrained('unused-local-target', directory, total_token=5, depth=2,
                                                  top_k=2, torch_dtype=torch.float32, device_map=None)
            self.assertTrue(torch.equal(restored.ea_layer.embed_tokens.weight, model.base_model.model.embed_tokens.weight))
            self.assertTrue(torch.equal(restored.ea_layer.lm_head.weight, model.ea_layer.lm_head.weight))

    def test_scheduling_skips_all_collection_cost(self):
        self.model.setup_online_adaptation(AdaptationConfig(update_every=2, validation_fraction=0, max_step_drift=0))
        self.model.eagenerate(self.prompt, max_new_tokens=6, max_length=40)
        _, _, _, stats = self.model.eagenerate(self.prompt, max_new_tokens=6, max_length=40, log=True)
        self.assertEqual(stats['adaptation']['status'], 'scheduled_skip')
        self.assertNotIn('teacher_collection', stats['stage_calls'])


class ExperimentTests(unittest.TestCase):
    def test_saved_tiny_models_end_to_end_evaluator_and_paired_report(self):
        from transformers import PreTrainedTokenizerFast
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        model = tiny_model()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target, draft = root / 'target', root / 'draft'
            model.base_model.save_pretrained(target)
            model.ea_layer.config.save_pretrained(draft)
            torch.save(model.ea_layer.state_dict(), draft / 'pytorch_model.bin')
            vocab = {'[UNK]': 0, **{f'w{i}': i for i in range(1, 32)}}
            raw = Tokenizer(WordLevel(vocab, unk_token='[UNK]'))
            raw.pre_tokenizer = Whitespace()
            tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw, unk_token='[UNK]')
            tokenizer.chat_template = "{% for m in messages %}{{ m['role'] + ' ' + m['content'] + ' ' }}{% endfor %}{% if add_generation_prompt %}assistant {% endif %}"
            tokenizer.save_pretrained(target)
            questions = root / 'questions.jsonl'
            questions.write_text('\n'.join(json.dumps({'question_id': i, 'category': 'tiny',
                'turns': [f'w{i+1} w4', 'w5 w6']}) for i in range(2)), encoding='utf8')
            common = ['--base-model-path', str(target), '--ea-model-path', str(draft),
                '--question-file', str(questions), '--max-new-tokens', '6', '--max-length', '80',
                '--total-token', '5', '--top-k', '2', '--depth', '2', '--dtype', 'float32',
                '--verify-greedy', '2', '--verify-tokens', '4', '--warmup-runs', '1', '--feature-chunk-size', '2']
            evaluate_main(common + ['--output-dir', str(root / 'baseline'), '--no-update'])
            evaluate_main(common + ['--output-dir', str(root / 'candidate'), '--freeze-after', '1'])
            report = compare(root / 'baseline', root / 'candidate')
            self.assertTrue(report['comparable_speed_claim'])
            self.assertEqual(report['paired_turns'], 4)
            frozen = compare(root / 'baseline', root / 'candidate', phase='frozen')
            self.assertEqual(frozen['paired_turns'], 2)
            stats = [json.loads(line) for line in (root / 'candidate' / 'stats.jsonl').read_text().splitlines()]
            self.assertEqual([r['adaptation_state']['weight_version'] for r in stats], [1, 2, 2, 2])
            with self.assertRaises(FileExistsError):
                evaluate_main(common + ['--output-dir', str(root / 'candidate')])
            manifest_path = root / 'candidate' / 'manifest.json'
            manifest = json.loads(manifest_path.read_text())
            manifest['args']['max_new_tokens'] += 1
            manifest_path.write_text(json.dumps(manifest), encoding='utf8')
            with self.assertRaises(ValueError):
                compare(root / 'baseline', root / 'candidate')

    def test_all_generated_commands_parse_and_sources_have_equal_caps(self):
        with tempfile.TemporaryDirectory() as directory:
            plan_main(['--base-model-path', '/target', '--ea-model-path', '/draft',
                       '--question-file', '/questions.jsonl', '--output-root', '/results',
                       '--plan-dir', directory, '--suite', 'all', '--order-seeds', '0,1'])
            jobs = json.loads((Path(directory) / 'plan.json').read_text(encoding='utf8'))['jobs']
            self.assertEqual(len(jobs), len(cases('all')) * 2)
            for job in jobs:
                args = parse_args(job['argv'][3:])
                if args.source == 'balanced':
                    self.assertEqual(args.max_prompt_positions + args.max_response_positions, 128)

    def test_question_shuffle_keeps_turns_and_rejects_duplicates(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'q.jsonl'
            questions = [{'question_id': i, 'turns': [str(i), 'follow-up']} for i in range(10)]
            path.write_text('\n'.join(json.dumps(q) for q in questions), encoding='utf8')
            a = load_questions(path, 2, 9, 42)
            b = load_questions(path, 2, 9, 42)
            self.assertEqual(a, b)
            self.assertEqual({q['question_id'] for q in a}, set(range(2, 9)))
            for q in a:
                self.assertEqual(q['turns'], [str(q['question_id']), 'follow-up'])
            path.write_text(json.dumps(questions[0]) + '\n' + json.dumps(questions[0]), encoding='utf8')
            with self.assertRaises(ValueError):
                load_questions(path)

    def test_summary_uses_ratio_of_sums_and_includes_update_time(self):
        rows = [dict(status='ok', new_tokens=n, total_steps=s, total_drafted_tokens=2*s,
                     total_accept_length=n-s, generation_seconds=t, stop_reason='eos',
                     adaptation={'status': 'updated'}) for n, s, t in [(10, 5, 1.), (90, 10, 9.)]]
        summary = summarize(rows)
        self.assertEqual(summary['tokens_per_second'], 10.)
        self.assertAlmostEqual(summary['returned_tokens_per_step'], 100/15)


if __name__ == '__main__':
    unittest.main()
