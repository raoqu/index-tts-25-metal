import array,json,math,sys
from pathlib import Path
roots=[Path(p) for p in sys.argv[1:3]]
manifests=[json.loads((p/'manifest.json').read_text()) for p in roots];results=[]
for a,b in zip(*(m['tensors'] for m in manifests)):
 assert a['name']==b['name'] and a['shape']==b['shape'] and a['dtype']==b['dtype']
 values=[]
 for root,t in zip(roots,[a,b]):
  with (root/'weights.bin').open('rb') as f:f.seek(t['offset']);raw=f.read(t['nbytes'])
  v=array.array('f' if t['dtype']=='f32' else 'I');v.frombytes(raw);values.append(v)
 errors=[float(x)-float(y) for x,y in zip(*values)];peak=max(map(abs,errors));rmse=math.sqrt(sum(x*x for x in errors)/len(errors))
 ok=peak<1e-4 and rmse<1e-5 and all(math.isfinite(x) for v in values for x in v)
 results.append(dict(name=a['name'],max_abs=peak,rmse=rmse,passed=ok))
print(json.dumps(dict(passed=all(r['passed'] for r in results),results=results),indent=2))
sys.exit(0 if all(r['passed'] for r in results) else 1)
