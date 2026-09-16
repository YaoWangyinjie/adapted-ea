import argparse,json,glob,statistics
from pathlib import Path

def load(p): return [json.loads(x) for x in open(p) if x.strip()]
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('paths',nargs='+'); ap.add_argument('--output',default='adaptation_data_report.json'); a=ap.parse_args(); report={}
 for p in a.paths:
  rows=load(p); report[Path(p).name]={'rows':len(rows),'categories':{},'status':{}}
  for r in rows:
   c=r.get('category','unknown'); report[Path(p).name]['categories'][c]=report[Path(p).name]['categories'].get(c,0)+1
   s=r.get('status','unknown'); report[Path(p).name]['status'][s]=report[Path(p).name]['status'].get(s,0)+1
 Path(a.output).write_text(json.dumps(report,indent=2,ensure_ascii=False)); print(json.dumps(report,indent=2))
if __name__=='__main__': main()
