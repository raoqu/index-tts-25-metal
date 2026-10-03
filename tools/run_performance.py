#!/usr/bin/env python3
"""Run binaries sequentially in ABBA order; never contend for the GPU."""
import argparse, json, subprocess
from pathlib import Path
from compare_performance import compare, read
p=argparse.ArgumentParser();p.add_argument('baseline');p.add_argument('candidate');p.add_argument('output');p.add_argument('--cases',default='tools/performance_cases.json');p.add_argument('--cycles',type=int,default=7);p.add_argument('--warmups',type=int,default=2);args=p.parse_args()
root=Path(args.output);root.mkdir(parents=True,exist_ok=True)
paths={'baseline':[],'candidate':[]}
for i,(label,binary) in enumerate([('baseline',args.baseline),('candidate',args.candidate),('candidate',args.candidate),('baseline',args.baseline)]):
    out=root/f'{i}-{label}';log=out.with_suffix('.jsonl')
    print(f'Running {label} round {i+1}/4',flush=True)
    with log.open('w') as f:subprocess.run([str(Path(binary).resolve()),'bundles/full','bundles/frontend',args.cases,str(out),str(args.cycles),str(args.warmups)],stdout=f,check=True)
    paths[label].append(log)
a=sum((read(f) for f in paths['baseline']),[]);b=sum((read(f) for f in paths['candidate']),[])
result=compare(a,b);result['rounds']=[compare(read(paths['baseline'][i]),read(paths['candidate'][i])) for i in range(2)]
result['baseline_binary']=args.baseline;result['candidate_binary']=args.candidate;result['cycles_per_round']=args.cycles
(root/'comparison.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'quality_pass':result['quality_pass'],'aggregate_improvement_percent':result['aggregate_improvement_percent'],'round_improvements':[r['aggregate_improvement_percent'] for r in result['rounds']]},indent=2),flush=True)
