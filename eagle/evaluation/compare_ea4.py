"""Paired comparison of completed EA4 runs, including costs and output equality."""
import argparse
import json
from pathlib import Path

from .gen_ea_answer_ea4 import summarize


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf8').splitlines() if line.strip()]


def read_run(directory):
    path = Path(directory)
    if not (path / 'summary.json').is_file():
        raise ValueError(f'Run is incomplete (no summary.json): {path}')
    manifest = json.loads((path / 'manifest.json').read_text(encoding='utf8'))
    rows = read_jsonl(path / 'stats.jsonl')
    keyed = {}
    for row in rows:
        key = (str(row['qid']), row['turn'])
        if key in keyed:
            raise ValueError(f'Duplicate turn {key} in {path}')
        keyed[key] = row
    answers = {(str(q['question_id']), turn): answer for q in read_jsonl(path / 'answers.jsonl')
               for turn, answer in enumerate(q['choices'][0]['turns'])}
    return manifest, keyed, answers


def compare(baseline, candidate, phase=None):
    bm, br, ba = read_run(baseline)
    cm, cr, ca = read_run(candidate)
    for name in ('base_model_path', 'ea_model_path', 'max_new_tokens', 'max_length', 'total_token',
                 'depth', 'top_k', 'dtype', 'seed', 'profile', 'verify_greedy', 'verify_tokens', 'warmup_runs'):
        if bm['args'][name] != cm['args'][name]:
            raise ValueError(f'Incomparable setting: {name}')
    for name in ('dataset_sha256', 'question_order', 'source_sha256', 'weight_hashes', 'torch', 'transformers', 'gpus'):
        if bm[name] != cm[name]:
            raise ValueError(f'Incomparable provenance: {name}')
    selected = {k for k, row in cr.items() if phase is None or row['phase'] == phase}
    keys = sorted(k for k in selected & br.keys() if br[k]['status'] == cr[k]['status'] == 'ok')
    base, adapted = summarize([br[k] for k in keys]), summarize([cr[k] for k in keys])
    speedup = base['generation_seconds'] / adapted['generation_seconds'] if adapted['generation_seconds'] else None
    output_mismatches = [list(k) for k in keys if k not in ba or k not in ca or ba[k] != ca[k]
                         or br[k]['output_token_sha256'] != cr[k]['output_token_sha256']]
    token_mismatches = [list(k) for k in keys if br[k]['new_tokens'] != cr[k]['new_tokens']]
    all_ok = len(keys) == len(selected) and (phase is not None or set(br) == set(cr))
    def difference(key):
        return adapted[key] - base[key] if adapted[key] is not None and base[key] is not None else None
    categories = {}
    for category in sorted({cr[k]['category'] for k in keys}):
        subset = [k for k in keys if cr[k]['category'] == category]
        categories[category] = {'baseline': summarize([br[k] for k in subset]),
                                'candidate': summarize([cr[k] for k in subset])}
    return {'paired_turns': len(keys), 'all_selected_turns_paired_ok': all_ok,
            'candidate_selected_turns': len(selected), 'baseline': base, 'candidate': adapted,
            'paired_walltime_speedup': speedup,
            'comparable_speed_claim': bool(keys) and all_ok and not output_mismatches and not token_mismatches,
            'acceptance_delta': difference('legacy_path_width_acceptance'),
            'returned_tokens_per_step_delta': difference('returned_tokens_per_step'),
            'output_mismatch_turns': output_mismatches, 'token_count_mismatch_turns': token_mismatches,
            'categories': categories,
            'note': 'Outputs compared by token-ID SHA256 and decoded answers. '
                    'Persistent questions are dependent; this report does not assume IID turns or compute a per-turn significance test.'}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline', required=True)
    p.add_argument('--candidate', required=True)
    p.add_argument('--phase', choices=['adaptation', 'frozen'])
    p.add_argument('--output')
    args = p.parse_args(argv)
    result = compare(args.baseline, args.candidate, args.phase)
    content = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)
    if args.output:
        with Path(args.output).open('x', encoding='utf8') as handle:
            handle.write(content + '\n')
    print(content)


if __name__ == '__main__':
    main()
