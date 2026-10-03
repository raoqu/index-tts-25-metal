#!/usr/bin/env python3
"""Replay captured real component inputs in persistent runtimes, comparing FP32."""
import argparse, array, json, math, subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('baseline');p.add_argument('candidate');p.add_argument('fixtures');p.add_argument('output');args=p.parse_args()
root=Path(args.output);root.mkdir(parents=True,exist_ok=True)
sequence=[('demo','gpt'),('demo','flow'),('qin','gpt'),('qin','flow'),('demo','flow'),('qin','vocode'),('demo','vocode')]
all_rows=[]
for label,binary in [('baseline',args.baseline),('candidate',args.candidate)]:
    lines=[]
    for i,(voice,op) in enumerate(sequence):
        source=Path(args.fixtures)/f'baseline-{voice}'/'inputs'/op
        lines.append(f'{op}\t{source}\t{root}/{label}-{i}.bin\n')
    proc=subprocess.run([str(Path(binary).resolve()),'--serve','bundles/full'],input=''.join(lines),text=True,capture_output=True,check=True)
    rows=[json.loads(line) for line in proc.stdout.splitlines()]
    if len(rows)!=len(sequence)+1 or not rows[0].get('ready') or not all(r.get('ok') for r in rows[1:]):raise RuntimeError(rows)
    all_rows.append(rows[1:])
results=[]
for i,(voice,op) in enumerate(sequence):
    a,b=all_rows[0][i],all_rows[1][i]
    assert a['dtype']==b['dtype'] and a['shape']==b['shape']
    if a['dtype']=='u32':
        exact=(root/f'baseline-{i}.bin').read_bytes()==(root/f'candidate-{i}.bin').read_bytes();ok=exact;peak=rmse=0 if exact else math.inf
    else:
        values=[]
        for label in ['baseline','candidate']:
            v=array.array('f');v.frombytes((root/f'{label}-{i}.bin').read_bytes());values.append(v)
        assert len(values[0])==len(values[1])
        errors=[float(x)-float(y) for x,y in zip(*values)]
        finite=all(math.isfinite(x) for v in values for x in v)
        peak=max(map(abs,errors));rmse=math.sqrt(sum(x*x for x in errors)/len(errors));ok=finite and peak<1e-4 and rmse<1e-5
    results.append({'case':f'{voice}-{op}-{i}','max_abs':peak,'rmse':rmse,'passed':ok})
result={'passed':all(r['passed'] for r in results),'results':results}
(root/'comparison.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
raise SystemExit(0 if result['passed'] else 1)
