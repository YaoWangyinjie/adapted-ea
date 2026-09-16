import argparse, json, os, time, tempfile
from pathlib import Path
import torch
from accelerate.utils import set_seed
from fastchat.llm_judge.common import load_questions
from tqdm import tqdm
from ..model.ea_model_2 import EaModel
from .gen_ea_answer_ds import _json_safe, _atomic_write_jsonl, _output_paths, _decode_output

def main():
 p=argparse.ArgumentParser()
 p.add_argument('--base-model-path',required=True); p.add_argument('--ea-model-path',required=True)
 p.add_argument('--question-file',default='eagle/data/mt_bench/question.jsonl'); p.add_argument('--answer-file',required=True)
 p.add_argument('--model-id',required=True); p.add_argument('--max-new-token',type=int,default=1024); p.add_argument('--max-length',type=int,default=2048); p.add_argument('--total-token',type=int,default=60); p.add_argument('--depth',type=int,default=5); p.add_argument('--top-k',type=int,default=10); p.add_argument('--temperature',type=float,default=0.0); p.add_argument('--adaptation-temperature',type=float,default=1.0); p.add_argument('--adaptation-lr',type=float,default=1e-5)
 p.add_argument('--scope',choices=['default','head_only','fc_only','midlayer_only','norm_only','lm_head_only'],default='head_only'); p.add_argument('--objective',choices=['kl','acceptance','acceptance_weighted_kl'],default='kl'); p.add_argument('--weight-mode',choices=['reset','persistent'],default='reset'); p.add_argument('--cache-scope',choices=['request','turn','choice','stream'],default='choice'); p.add_argument('--no-update',action='store_true'); p.add_argument('--experiment-name',required=True); p.add_argument('--max-relative-drift',type=float,default=.02); p.add_argument('--gradient-clip-norm',type=float,default=.5); p.add_argument('--anchor-weight',type=float,default=1e-3); p.add_argument('--question-begin',type=int); p.add_argument('--question-end',type=int); args=p.parse_args(); set_seed(0)
 model=EaModel.from_pretrained(base_model_path=args.base_model_path,ea_model_path=args.ea_model_path,total_token=args.total_token,depth=args.depth,top_k=args.top_k,torch_dtype=torch.float16,low_cpu_mem_usage=True,device_map='auto')
 if not args.no_update:
  model.setup_online_adaptation(adaptation_lr=args.adaptation_lr,adaptation_temperature=args.adaptation_temperature,mode=args.weight_mode,scope=args.scope,objective=args.objective,reset_granularity='choice' if args.cache_scope == 'request' else args.cache_scope)
  model.max_relative_drift=args.max_relative_drift; model.gradient_clip_norm=args.gradient_clip_norm; model.anchor_weight=args.anchor_weight
 tok=model.get_tokenizer(); qs=load_questions(args.question_file,args.question_begin,args.question_end); answers=[]; stats=[]; failures=[]
 for dataset_index, q in enumerate(tqdm(qs), start=args.question_begin or 0):
  probe_messages=[]
  oversized_turns=[]
  for probe_turn, probe_text in enumerate(q.get('turns', [])):
   probe_messages.append({'role':'user','content':probe_text})
   probe_prompt=tok.apply_chat_template(probe_messages,tokenize=False,add_generation_prompt=True)
   probe_len=len(tok([probe_prompt],add_special_tokens=False).input_ids[0])
   if probe_len >= args.max_length:
    oversized_turns.append((probe_turn, probe_len))
   probe_messages.append({'role':'assistant','content':''})
  if oversized_turns:
   for failed_turn, prompt_tokens in oversized_turns:
    failure={'qid':q['question_id'],'dataset_index':dataset_index,'category':q.get('category','unknown'),'turn':failed_turn,'prompt_tokens':prompt_tokens,'max_length':args.max_length,'reason':'prompt_exceeds_cache_capacity','status':'skipped','experiment':args.experiment_name}
    failures.append(_json_safe(failure)); stats.append(_json_safe(failure))
   continue
  choices=[]
  question_failed=False
  if args.weight_mode=='reset' and args.cache_scope in ('choice','turn'): model.reset_online_adaptation()
  messages=[]; turns=[]; idxs=[]; nts=[]; times=[]
  for turn,text in enumerate(q.get('turns',[])):
   if turn and args.cache_scope=='turn': model.reset_online_adaptation()
   messages.append({'role':'user','content':text}); prompt=tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True); ids=tok([prompt],add_special_tokens=False).input_ids; start=time.time()
   try:
    out,n,idx,run=model.eagenerate(torch.as_tensor(ids).cuda(),temperature=args.temperature,max_new_tokens=args.max_new_token,max_length=args.max_length,log=True,is_llama3=True,enable_adaptation=not args.no_update,adaptation_lr=args.adaptation_lr,adaptation_temperature=args.adaptation_temperature,diagnostics=True)
   except RuntimeError as exc:
    failure={'qid':q['question_id'],'dataset_index':dataset_index,'category':q.get('category','unknown'),'turn':turn,'prompt_tokens':len(ids[0]),'max_length':args.max_length,'reason':'runtime_error','error':str(exc),'status':'skipped','experiment':args.experiment_name}
    failures.append(_json_safe(failure)); stats.append(_json_safe(failure)); question_failed=True; break
   elapsed=time.time()-start
   diagnostics=run.get('diagnostics', {})
   if diagnostics.get('cache_stop'):
    failure={'qid':q['question_id'],'dataset_index':dataset_index,'category':q.get('category','unknown'),'turn':turn,'prompt_tokens':len(ids[0]),'max_length':args.max_length,'reason':diagnostics['cache_stop'],'status':'skipped','experiment':args.experiment_name}
    failures.append(_json_safe(failure)); stats.append(_json_safe(failure)); question_failed=True; break
   stat={'qid':q['question_id'],'category':q.get('category','unknown'),'turn':turn,'choice':0,'experiment':args.experiment_name,'scope':args.scope,'objective':args.objective,'weight_mode':args.weight_mode,'cache_scope':args.cache_scope,'new_tokens':int(n),'time':elapsed,**{k:run.get(k) for k in ('total_accept_length','total_drafted_tokens','total_steps','losses','diagnostics')}}; stats.append(_json_safe(stat)); answer=_decode_output(tok,out,len(ids[0])); turns.append(answer); idxs.append(int(idx)); nts.append(int(n)); times.append(elapsed); messages.append({'role':'assistant','content':answer})
  if question_failed:
   continue
  choices.append({'index':0,'turns':turns,'idxs':idxs,'new_tokens':nts,'wall_time':times}); answers.append({'question_id':q['question_id'],'answer_id':str(q['question_id'])+'-'+str(time.time_ns()),'model_id':args.model_id,'choices':choices,'tstamp':time.time()})
 answer,statfile,machine=_output_paths(args.answer_file); _atomic_write_jsonl(answer,answers); _atomic_write_jsonl(machine,stats); failure_path=Path(args.answer_file).with_name(Path(args.answer_file).stem+'_failures.jsonl'); _atomic_write_jsonl(str(failure_path),failures); Path(statfile).write_text(json.dumps({'experiment':args.experiment_name,'rows':len(stats),'failures':len(failures)},indent=2)); print('Saved',answer,machine,'failures',failure_path)
if __name__=='__main__': main()
