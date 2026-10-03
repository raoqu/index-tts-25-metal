#!/usr/bin/env python3
"""Compare benchmark JSONL and PCM16 audio; no third-party dependencies."""
import argparse, array, json, math, statistics, sys, wave
from pathlib import Path

def read(path):
    rows=[json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]
    return [r for r in rows if r.get('kind')=='synthesis' and not r['warmup']]

def compare(a,b):
    result={}; passed=True
    for name in sorted({r['case'] for r in a}):
        aa=[r for r in a if r['case']==name];bb=[r for r in b if r['case']==name]
        x=statistics.median(r['total_seconds'] for r in aa);y=statistics.median(r['total_seconds'] for r in bb)
        codes=lambda r:[(s['text_ids'],s['codes'],s['mel_frames']) for s in r['segments']]
        exact=all(codes(r)==codes(aa[0]) for r in aa+bb)
        samples=[]
        for r in (aa[0],bb[0]):
            with wave.open(r['output'],'rb') as w:
                assert w.getsampwidth()==2 and w.getnchannels()==1
                samples.append(array.array('h',w.readframes(w.getnframes())))
        equal_length=len(samples[0])==len(samples[1]);peak=rmse=float('inf')
        if equal_length:
            errors=[(x-y)/32767 for x,y in zip(*samples)]
            peak=max(map(abs,errors));rmse=math.sqrt(sum(v*v for v in errors)/len(errors))
        stable=len({codes_hash(r) for r in aa+bb})==1
        repeated=all(len({r["wav_sha256"] for r in rows})==1 for rows in (aa,bb))
        ok=exact and equal_length and peak<1e-4 and rmse<1e-5 and stable and repeated
        passed &= ok
        result[name]={'baseline_seconds':x,'candidate_seconds':y,'improvement_percent':100*(x-y)/x,'codes_exact':exact,'pcm_max_abs':peak,'pcm_rmse':rmse,'quality_pass':ok,
            'baseline_stages':{k:statistics.median(r['stage_seconds'][k] for r in aa) for k in aa[0]['stage_seconds']},
            'candidate_stages':{k:statistics.median(r['stage_seconds'][k] for r in bb) for k in bb[0]['stage_seconds']},
            'baseline_memory_first':aa[0]['memory'],'baseline_memory_last':aa[-1]['memory'],
            'candidate_memory_first':bb[0]['memory'],'candidate_memory_last':bb[-1]['memory'],
            'candidate_metal_allocations': [r['metal']['allocations'] for r in bb]}
    total_a=sum(v['baseline_seconds'] for v in result.values());total_b=sum(v['candidate_seconds'] for v in result.values())
    return {'quality_pass':passed,'aggregate_improvement_percent':100*(total_a-total_b)/total_a,'cases':result}

def codes_hash(r):return json.dumps(r['segments'],sort_keys=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('baseline');p.add_argument('candidate');p.add_argument('--output');args=p.parse_args()
    result=compare(read(args.baseline),read(args.candidate));out=json.dumps(result,ensure_ascii=False,indent=2)
    if args.output:Path(args.output).write_text(out+'\n')
    print(out);sys.exit(0 if result['quality_pass'] else 1)
