"""Sequential, auditable EA4 evaluator. Run with python -m from the repo root."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import random
import subprocess


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-model-path', required=True)
    p.add_argument('--ea-model-path', required=True)
    p.add_argument('--question-file', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--experiment-name', default='ea4')
    p.add_argument('--no-update', action='store_true')
    p.add_argument('--weight-mode', choices=['persistent', 'reset'], default='persistent')
    p.add_argument('--source', choices=['prompt', 'response', 'balanced'], default='response')
    p.add_argument('--update-at', choices=['prefill', 'warmup', 'end'], default='end')
    p.add_argument('--scope', choices=['head_only', 'norm_only', 'lm_head_only'], default='head_only')
    p.add_argument('--precision', choices=['master_fp32', 'roundtrip'], default='master_fp32')
    p.add_argument('--alignment', choices=['correct', 'legacy'], default='correct')
    p.add_argument('--learning-rate', type=float, default=1e-6)
    p.add_argument('--distill-temperature', type=float, default=1.0)
    p.add_argument('--prompt-weight', type=float, default=.25)
    p.add_argument('--max-prompt-positions', type=int, default=128)
    p.add_argument('--max-response-positions', type=int, default=128)
    p.add_argument('--warmup-steps', type=int, default=5)
    p.add_argument('--update-every', type=int, default=1, help='Every K generation turns; first turn updates')
    p.add_argument('--train-steps', type=int, default=1)
    p.add_argument('--chunk-size', type=int, default=32)
    p.add_argument('--feature-chunk-size', type=int, default=256)
    p.add_argument('--validation-fraction', type=float, default=.2)
    p.add_argument('--gradient-clip', type=float, default=.2)
    p.add_argument('--max-step-drift', type=float, default=.02)
    p.add_argument('--max-anchor-drift', type=float, default=0)
    p.add_argument('--anchor-weight', type=float, default=0)
    p.add_argument('--replay-mib', type=float, default=0)
    p.add_argument('--replay-positions', type=int, default=64)
    p.add_argument('--freeze-after', type=int, help='Adapt on first N stream questions, then freeze; report both phases')
    p.add_argument('--max-new-tokens', type=int, default=1024)
    p.add_argument('--max-length', type=int, default=4096)
    p.add_argument('--total-token', type=int, default=60)
    p.add_argument('--depth', type=int, default=5)
    p.add_argument('--top-k', type=int, default=10)
    p.add_argument('--dtype', choices=['float16', 'bfloat16', 'float32'], default='float16')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--order-seed', type=int, help='Shuffle whole questions, never their internal turns')
    p.add_argument('--question-begin', type=int, default=0)
    p.add_argument('--question-end', type=int)
    p.add_argument('--profile', action='store_true', help='Synchronize CUDA around each stage; use separate profiling runs')
    p.add_argument('--verify-greedy', type=int, default=0, help='Reference-check first N questions (outside generation timing)')
    p.add_argument('--verify-tokens', type=int, default=64)
    p.add_argument('--warmup-runs', type=int, default=2, help='Unmeasured inference-only runs; never update weights')
    p.add_argument('--hash-weights', action='store_true', help='Hash all local weight shards once; can be slow')
    return p.parse_args(argv)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_questions(path, begin=0, end=None, order_seed=None):
    records = []
    seen = set()
    for line in Path(path).read_text(encoding='utf-8-sig').splitlines():
        if not line.strip():
            continue
        q = json.loads(line)
        key = str(q['question_id'])
        if key in seen:
            raise ValueError(f'Duplicate question_id: {key}')
        seen.add(key)
        if not q.get('turns') or any(not isinstance(t, str) for t in q['turns']):
            raise ValueError(f'Invalid turns for question {key}')
        records.append(dict(q, original_index=len(records)))
    records = records[begin:end]
    if order_seed is not None:
        random.Random(order_seed).shuffle(records)
    return records


def write_line(handle, value):
    handle.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')
    handle.flush()  # prior completed turns survive a later process/model failure


def summarize(rows):
    valid = [r for r in rows if r.get('status') == 'ok']
    def total(key):
        return sum(r.get(key, 0) for r in valid)
    steps, widths, seconds = total('total_steps'), total('total_drafted_tokens'), total('generation_seconds')
    return {'turns': len(valid), 'failed_or_skipped': len(rows) - len(valid),
            'new_tokens': total('new_tokens'), 'generation_seconds': seconds,
            'total_steps': steps,
            'total_accept_length': total('total_accept_length'),
            'verified_tokens_per_step_including_root': (total('total_accept_length') + steps) / steps if steps else None,
            'discarded_verified_tokens': total('total_accept_length') + steps - total('new_tokens'),
            'length_aggregation': 'pooled verification steps; accepted_children_per_step excludes root',
            'tokens_per_second': total('new_tokens') / seconds if seconds else None,
            'legacy_path_width_acceptance': total('total_accept_length') / widths if widths else None,
            'accepted_children_per_step': total('total_accept_length') / steps if steps else None,
            'returned_tokens_per_step': total('new_tokens') / steps if steps else None,
            'eos_turns': sum(r['stop_reason'] == 'eos' for r in valid),
            'limit_turns': sum(r['stop_reason'] != 'eos' for r in valid),
            'updated_turns': sum(r['adaptation']['status'] == 'updated' for r in valid),
            'rolled_back_turns': sum(r['adaptation']['status'] == 'rolled_back' for r in valid)}


def main(argv=None):
    args = parse_args(argv)
    import torch
    import transformers
    from ..model.ea_model_4 import EaModel
    from ..model.online_head import AdaptationConfig

    if args.freeze_after is not None and (args.freeze_after < 0 or args.weight_mode != 'persistent'):
        raise ValueError('freeze-after requires persistent mode and a nonnegative boundary')
    if args.verify_greedy < 0 or args.verify_tokens < 1:
        raise ValueError('Invalid reference verification settings')
    if args.question_begin < 0 or args.max_new_tokens < 1 or args.max_length < 4 or args.warmup_runs < 0:
        raise ValueError('Invalid question slice, generation limit, or warmup count')
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Refusing to mix with existing results: {output}')
    questions = load_questions(args.question_file, args.question_begin, args.question_end, args.order_seed)
    if not questions:
        raise ValueError('No questions selected')
    config = AdaptationConfig(learning_rate=args.learning_rate, temperature=args.distill_temperature,
        scope=args.scope, source=args.source, prompt_weight=args.prompt_weight,
        max_prompt_positions=args.max_prompt_positions, max_response_positions=args.max_response_positions,
        update_at=args.update_at, warmup_steps=args.warmup_steps, update_every=args.update_every,
        train_steps=args.train_steps, chunk_size=args.chunk_size, feature_chunk_size=args.feature_chunk_size,
        validation_fraction=args.validation_fraction,
        gradient_clip=args.gradient_clip, max_step_drift=args.max_step_drift,
        max_anchor_drift=args.max_anchor_drift, anchor_weight=args.anchor_weight,
        replay_bytes=int(args.replay_mib * 1024 ** 2), replay_positions=args.replay_positions,
        precision=args.precision, alignment=args.alignment, seed=args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    model = EaModel.from_pretrained(args.base_model_path, args.ea_model_path,
        total_token=args.total_token, depth=args.depth, top_k=args.top_k,
        torch_dtype=getattr(torch, args.dtype), low_cpu_mem_usage=True)
    if not args.no_update:
        model.setup_online_adaptation(config)
    if args.warmup_runs:
        prompt = model.tokenizer.apply_chat_template([{'role': 'user', 'content': 'Calculate 2 + 3.'}],
                                                     tokenize=False, add_generation_prompt=True)
        ids = model.tokenizer(prompt, return_tensors='pt', add_special_tokens=False).input_ids.to(model.input_device)
        for _ in range(args.warmup_runs):
            model.eagenerate(ids, max_new_tokens=min(16, args.max_new_tokens), max_length=args.max_length,
                             enable_adaptation=False)
        model.calls = 0  # warmup must not change adaptation scheduling or reservoir seeds
    root = Path(__file__).resolve().parents[2]
    commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=root, capture_output=True, text=True)
    sources = ['eagle/model/ea_model_4.py', 'eagle/model/online_head.py', 'eagle/model/cnets.py',
               'eagle/model/modeling_llama_kv.py', 'eagle/model/utils.py', 'eagle/model/kv_cache.py',
               'eagle/model/configs.py', 'eagle/evaluation/gen_ea_answer_ea4.py']
    manifest = {'args': vars(args), 'adaptation': asdict(config), 'git_commit': commit.stdout.strip() if commit.returncode == 0 else None,
        'source_sha256': {p: sha256(root / p) for p in sources}, 'dataset_sha256': sha256(args.question_file),
        'question_order': [q['question_id'] for q in questions], 'torch': torch.__version__,
        'transformers': transformers.__version__, 'python': platform.python_version(), 'platform': platform.platform(),
        'cuda': torch.version.cuda, 'gpus': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        'target_config': model.config.to_dict(), 'hidden_layer_indices': model.hidden_layer_indices,
        'weight_hashes': {}}
    for directory in (args.base_model_path, args.ea_model_path):
        if Path(directory).is_dir():
            for path in Path(directory).iterdir():
                if path.name in ('config.json', 'tokenizer_config.json') or (args.hash_weights and path.suffix in ('.bin', '.safetensors')):
                    manifest['weight_hashes'][str(path)] = sha256(path)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf8')
    rows = []
    with (output / 'stats.jsonl').open('w', encoding='utf8') as stats_file, (output / 'answers.jsonl').open('w', encoding='utf8') as answers_file:
        for stream_index, question in enumerate(questions):
            if args.weight_mode == 'reset' and model.learner is not None:
                model.reset_online_adaptation()
            active = not args.no_update and (args.freeze_after is None or stream_index < args.freeze_after)
            messages, answers = [], []
            for turn, text in enumerate(question['turns']):
                messages.append({'role': 'user', 'content': text})
                prompt = model.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                ids = model.tokenizer(prompt, return_tensors='pt', add_special_tokens=False).input_ids.to(model.input_device)
                meta = {'qid': question['question_id'], 'turn': turn, 'choice': 0, 'stream_index': stream_index,
                        'original_index': question['original_index'], 'category': question.get('category', 'unknown'),
                        'prompt_tokens': ids.shape[1], 'phase': 'adaptation' if active else 'frozen',
                        'experiment': args.experiment_name, 'version_before_generation': model.get_adaptation_state()['weight_version']}
                if ids.shape[1] >= args.max_length:
                    row = dict(meta, status='skipped', reason='prompt_exceeds_context_capacity')
                    write_line(stats_file, row); rows.append(row)
                    break
                try:
                    reference = None
                    if stream_index < args.verify_greedy:
                        reference = model.naivegenerate(ids, min(args.verify_tokens, args.max_new_tokens), args.max_length)
                    out, n, idx, stats = model.eagenerate(ids, max_new_tokens=args.max_new_tokens,
                        max_length=args.max_length, log=True, enable_adaptation=active, profile=args.profile)
                    if reference is not None:
                        matched = torch.equal(out[:, :reference.shape[1]], reference)
                        if reference[0, -1].item() in model._stops(True) and out.shape[1] != reference.shape[1]:
                            matched = False
                        if not matched:
                            raise AssertionError('EA4 greedy output differs from target reference; stop this experiment')
                    token_bytes = json.dumps(out[0, ids.shape[1]:].tolist(), separators=(',', ':')).encode('ascii')
                    row = dict(meta, **stats, status='ok', greedy_reference_checked=reference is not None,
                               output_token_sha256=hashlib.sha256(token_bytes).hexdigest())
                    answer = model.tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
                    write_line(stats_file, row); rows.append(row)
                    answers.append(answer)
                    messages.append({'role': 'assistant', 'content': answer})
                    print(f"{stream_index + 1}/{len(questions)} q={question['question_id']} turn={turn} tokens={n} "
                          f"update={stats['adaptation']['status']} version={stats['adaptation_state']['weight_version']}", flush=True)
                except Exception as exc:
                    write_line(stats_file, dict(meta, status='error', error_type=type(exc).__name__, error=str(exc)))
                    raise  # never silently continue a possibly invalid persistent trajectory
            write_line(answers_file, {'question_id': question['question_id'], 'model_id': args.experiment_name,
                'complete': len(answers) == len(question['turns']), 'choices': [{'index': 0, 'turns': answers}]})
    report = {'overall': summarize(rows), 'phases': {phase: summarize([r for r in rows if r['phase'] == phase])
              for phase in ('adaptation', 'frozen')}, 'metric_note': 'acceptance uses legacy path-width denominator; throughput includes end-of-answer updates'}
    (output / 'summary.json').write_text(json.dumps(report, indent=2), encoding='utf8')


if __name__ == '__main__':
    main()
