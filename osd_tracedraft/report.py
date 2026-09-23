"""Aggregate raw counters and compare only matched runs."""
import argparse
import csv
from pathlib import Path
from .io_utils import read_json,read_jsonl,write_json,token_hash
from .metrics import summarize,gains


def aggregate(run_dir):
    root=Path(run_dir);config=read_json(root/'config.json');state=read_json(root/'state.json')
    rows=[];training=0;block_reports=[];osd_reports=[]
    for block in state['blocks']:
        if block.get('status')!='complete': continue
        inf=root/block['infer_directory'];batch=read_jsonl(inf/'turns.jsonl');rows.extend(batch)
        seconds=0
        if block.get('train_directory'):
            training_report=read_json(root/block['train_directory']/'report.json')
            osd_reports.append(training_report)
            seconds=training_report['stage_seconds']
            training+=seconds
        block_reports.append(dict(block=block['index'],**summarize(batch,seconds)))
    summary=summarize(rows,training,state.get('pipeline_seconds'))
    summary.update(status=state['status'],method=config['method'],model_profile=config['model_profile'],
        osd_optimizer_steps=sum(r.get('steps',0) for r in osd_reports),
        osd_attempted_steps=sum(r.get('attempted_steps',r.get('steps',0)) for r in osd_reports),
        osd_rejected_blocks=sum(r.get('guard',{}).get('accepted') is False for r in osd_reports),
        request_count=len({r['stream_index'] for r in rows}),prompt_truncated_turns=sum(r['prompt_tokens_removed']>0 for r in rows),
        generated_tokens_sha256=token_hash([(r['question_id'],r['turn'],r['generated_sha256']) for r in rows]),
        prompts_sha256=token_hash([(r['question_id'],r['turn'],r['prompt_sha256']) for r in rows]))
    write_json(root/'summary.json',summary);write_json(root/'block_summaries.json',block_reports)
    windows=[]
    for start in range(0,1+max((r['stream_index'] for r in rows),default=-1),100):
        subset=[r for r in rows if start<=r['stream_index']<start+100]
        windows.append(dict(request_start=start+1,request_end=max(r['stream_index'] for r in subset)+1,
                            **summarize(subset)))
    # Windows intentionally report inference only: block-training costs are in
    # block_summaries and whole-run summary, not spread arbitrarily over queries.
    write_json(root/'windows_100.json',windows)
    return summary


MATCH_KEYS=('model_profile','target','draft','seed','order_seed','max_new_tokens','max_length','dtype',
            'total_token','depth','top_k','prompt_truncation','warmup_runs','verify_greedy','verify_tokens','osd_block_size')


def compare_runs(run_dirs,output,reference_method='baseline'):
    if reference_method not in ('baseline','osd'):
        raise ValueError('Reference must be frozen EAGLE-3 (baseline) or OSD-EAGLE-3 (osd)')
    runs=[]
    for path in run_dirs:
        p=Path(path);summary=aggregate(p)
        if summary['status']!='complete': raise ValueError(f'Incomplete run: {p}')
        runs.append((p,read_json(p/'config.json'),read_json(p/'manifest.json'),summary))
    baselines=[r for r in runs if r[1]['method']==reference_method]
    if len(baselines)!=1: raise ValueError('Provide exactly one matched baseline per comparison')
    base=baselines[0];result=[]
    for p,c,m,s in runs:
        mismatch=[k for k in MATCH_KEYS if c[k]!=base[1][k]]
        for k,default in (('target_quantization','none'),('draft_device','cuda'),('cpu_threads',0),('output_norm_fp32',False)):
            if c.get(k,default)!=base[1].get(k,default): mismatch.append(k)
        if m['questions_sha256']!=base[2]['questions_sha256']: mismatch.append('dataset')
        for k in ('target_files','draft_files','source_files'):
            if m[k]!=base[2][k]: mismatch.append(k)
        if mismatch: raise ValueError(f'{p}: unmatched settings/checkpoints/source: {mismatch}')
        if s['prompts_sha256']!=base[3]['prompts_sha256'] or s['generated_tokens_sha256']!=base[3]['generated_tokens_sha256']:
            raise ValueError(f'{p}: greedy token sequences differ; inspect answers before calculating speedup')
        row=dict(run_directory=str(p.resolve()),method=c['method'],**{k:v for k,v in s.items() if k!='method'},
                 **gains(base[3],s));result.append(row)
    osd=next((r for r in runs if r[1]['method']=='osd'),None)
    combined=next((r for r in runs if r[1]['method']=='osd_tracedraft'),None)
    extra=None;slow_checkpoints=None
    if osd and combined:
        osd_keys=[k for k in osd[1] if k.startswith('osd_')]+['train_last_block']
        if any(osd[1][k]!=combined[1][k] for k in osd_keys): raise ValueError('OSD hyperparameters differ across combination controls')
        extra=gains(osd[3],combined[3])
        def checkpoint_hashes(run):
            state=read_json(run[0]/'state.json')
            return {str(b['index']):read_json(run[0]/b['train_directory']/'checkpoint/provenance.json')['weights_sha256']
                    for b in state['blocks'] if b.get('train_directory')}
        left=checkpoint_hashes(osd);right=checkpoint_hashes(combined)
        slow_checkpoints=dict(osd=left,combined=right,identical=left==right)
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    write_json(out/'comparison.json',dict(reference_method=reference_method,runs=result,combined_vs_osd=extra,slow_checkpoint_comparison=slow_checkpoints))
    keys=['method','acceptance_percent','average_length','average_length_including_root','returned_tokens_per_step',
          'new_tokens','tokens_per_second_generation','speedup_generation','tokens_per_second_training_inclusive',
          'speedup_training_inclusive','tokens_per_second_pipeline','speedup_pipeline','acceptance_delta_pp',
          'average_length_relative_gain_percent','osd_training_seconds','osd_optimizer_steps','osd_attempted_steps',
          'osd_rejected_blocks','trace_updated_turns','trace_rolled_back_turns',
          'trace_validation_rejected_turns','trace_skipped_turns']
    with (out/'comparison.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore');writer.writeheader();writer.writerows(result)
    def fmt(x,d=3): return f'{x:.{d}f}' if isinstance(x,(int,float)) else '--'
    lines=['# EAGLE-3 / OSD / TraceDraft comparison','',
        f"Target model: {base[1]['target']}",'',
        '| Method | Acceptance (%) | Length (children) | Length (+ root) | Generation tok/s | Generation ratio | Training-inclusive tok/s | Training-inclusive ratio |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in result:
        lines.append('| '+ ' | '.join([r['method']]+[fmt(r[k]) for k in ('acceptance_percent','average_length',
            'average_length_including_root','tokens_per_second_generation','speedup_generation',
            'tokens_per_second_training_inclusive','speedup_training_inclusive')])+' |')
    reference_label='frozen EAGLE-3' if reference_method=='baseline' else 'OSD-EAGLE-3'
    lines+=['',f'Ratios use the matched {reference_label} reference (1.000). They are not speedups over autoregressive decoding.',
        'Generation time includes TraceDraft collection and updates. Training-inclusive time additionally charges conversation resets and block OSD training/checkpoint saving.',
        'Pipeline time also includes model reloads, warmup, correctness probes, tokenization, disk I/O, and process startup; see CSV/JSON.',
        'Acceptance follows local EA4: accepted children / sum of candidate path widths (which include the root). Average length excludes the root; +root is before final truncation.',
        'All matched runs generated identical token sequences.']
    if extra:
        lines+=['',f"OSD+TraceDraft / OSD: generation ratio {fmt(extra['speedup_generation'])}; training-inclusive ratio {fmt(extra['speedup_training_inclusive'])}."]
    (out/'comparison.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--runs',nargs='+',required=True);p.add_argument('--output')
    p.add_argument('--reference-method',choices=('baseline','osd'),default='baseline')
    args=p.parse_args()
    if args.output: compare_runs(args.runs,args.output,args.reference_method)
    else:
        for path in args.runs: print(aggregate(path))
if __name__=='__main__': main()
