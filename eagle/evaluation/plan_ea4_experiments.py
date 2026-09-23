"""Write reviewable EA4 experiment commands. Does not load models or run jobs."""
import argparse
import json
from pathlib import Path
import shlex


def cases(suite, freeze_after=32):
    base = {'baseline': ['--no-update'], 'end_response': []}
    if suite == 'smoke':
        return base
    core = {
        **base,
        'end_prompt': ['--source', 'prompt'],
        'end_balanced': ['--source', 'balanced', '--max-prompt-positions', '64', '--max-response-positions', '64'],
        'end_response_legacy': ['--alignment', 'legacy'],
        'end_response_roundtrip': ['--precision', 'roundtrip'],
        'prefill_prompt_persistent': ['--update-at', 'prefill', '--source', 'prompt'],
        'prefill_prompt_reset': ['--update-at', 'prefill', '--source', 'prompt', '--weight-mode', 'reset'],
        'warmup_balanced_persistent': ['--update-at', 'warmup', '--source', 'balanced', '--max-prompt-positions', '64', '--max-response-positions', '64'],
        'warmup_balanced_reset': ['--update-at', 'warmup', '--source', 'balanced', '--max-prompt-positions', '64', '--max-response-positions', '64', '--weight-mode', 'reset'],
        'end_response_every8': ['--update-every', '8'],
        'end_response_every32': ['--update-every', '32'],
    }
    stability = {
        **base,
        'end_response_lr3e-7': ['--learning-rate', '3e-7'],
        'end_response_lr3e-6': ['--learning-rate', '3e-6'],
        'end_response_lr1e-5': ['--learning-rate', '1e-5'],
        'end_response_norm_only': ['--scope', 'norm_only'],
        'end_response_replay': ['--replay-mib', '64', '--replay-positions', '64'],
        'end_response_anchor_guard': ['--max-anchor-drift', '.02'],
        'end_response_anchor_penalty': ['--anchor-weight', '.01'],
        'end_response_freeze': ['--freeze-after', str(freeze_after)],
    }
    return core if suite == 'core' else stability if suite == 'stability' else {**core, **stability}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-model-path', required=True)
    p.add_argument('--ea-model-path', required=True)
    p.add_argument('--question-file', required=True)
    p.add_argument('--output-root', required=True)
    p.add_argument('--plan-dir', required=True)
    p.add_argument('--suite', choices=['smoke', 'core', 'stability', 'all'], default='smoke')
    p.add_argument('--order-seeds', default='0', help='Comma-separated order seeds; e.g. 0,1,2')
    p.add_argument('--max-new-tokens', type=int, default=1024)
    p.add_argument('--max-length', type=int, default=4096)
    p.add_argument('--question-begin', type=int, default=0)
    p.add_argument('--question-end', type=int)
    p.add_argument('--freeze-after', type=int, default=32)
    p.add_argument('--python', default='python')
    args = p.parse_args(argv)
    seeds = list(dict.fromkeys(int(s) for s in args.order_seeds.split(',')))
    destination = Path(args.plan_dir)
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f'Refusing to overwrite plan: {destination}')
    jobs = []
    for seed in seeds:
        for name, extra in cases(args.suite, args.freeze_after).items():
            name = f'{name}_order{seed}'
            command = [args.python, '-m', 'eagle.evaluation.gen_ea_answer_ea4',
                '--base-model-path', args.base_model_path, '--ea-model-path', args.ea_model_path,
                '--question-file', args.question_file, '--output-dir', str(Path(args.output_root) / name),
                '--experiment-name', name, '--order-seed', str(seed),
                '--max-new-tokens', str(args.max_new_tokens), '--max-length', str(args.max_length),
                '--question-begin', str(args.question_begin)]
            end = args.question_end
            if args.suite == 'smoke':
                end = min(end, args.question_begin + 4) if end is not None else args.question_begin + 4
                command += ['--verify-greedy', '4', '--verify-tokens', '32']
            if end is not None:
                command += ['--question-end', str(end)]
            command += extra
            jobs.append({'name': name, 'argv': command})
    destination.mkdir(parents=True, exist_ok=True)
    (destination / 'plan.json').write_text(json.dumps({'suite': args.suite, 'jobs': jobs}, indent=2, ensure_ascii=False), encoding='utf8')
    (destination / 'run.sh').write_text('#!/usr/bin/env bash\nset -euo pipefail\n# Run from the adapted-ea repository root.\n'
                                       + '\n'.join(shlex.join(j['argv']) for j in jobs) + '\n', encoding='utf8')
    def ps_quote(arg):
        return "'" + arg.replace("'", "''") + "'"
    (destination / 'run.ps1').write_text("$ErrorActionPreference = 'Stop'\n# Run from the adapted-ea repository root.\n"
        + '\n'.join('& ' + ' '.join(ps_quote(s) for s in j['argv']) + '\nif ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }' for j in jobs) + '\n', encoding='utf-8-sig')
    print(f'Wrote {len(jobs)} commands to {destination}; no experiments have been started.')


if __name__ == '__main__':
    main()
