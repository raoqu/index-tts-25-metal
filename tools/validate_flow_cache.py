import array,json,math,struct,subprocess,sys
from pathlib import Path
baseline,candidate,fixture,outdir=sys.argv[1:];source=Path(fixture);root=Path(outdir);root.mkdir(parents=True,exist_ok=True)
man=json.loads((source/'manifest.json').read_text());tensors={t['name']:t for t in man['tensors']}
for steps in [25,*range(1,10),201]:
 dest=root/f'input-{steps}';dest.mkdir(exist_ok=True);new=[]
 with (dest/'weights.bin').open('wb') as f:
  f.write(struct.pack('<III',0x3254494d,1,4096))
  for name,old in [('noise','cfm.noise'),('prompt','cfm.prompt'),('condition','cfm.condition'),('style','cfm.style'),('config',None)]:
   if old:
    t=dict(tensors[old]);
    with (source/man['weights_file']).open('rb') as g:g.seek(t['offset']);data=g.read(t['nbytes'])
   else:t=dict(shape=[2],dtype='f32');data=struct.pack('<ff',steps,.7)
   offset=(f.tell()+4095)//4096*4096;f.write(bytes(offset-f.tell()));f.write(data);t.update(name=name,offset=offset,nbytes=len(data),layout='row_major',component='input');new.append(t)
 (dest/'manifest.json').write_text(json.dumps(dict(format='MIT2',version=1,alignment=4096,endianness='little',weights_file='weights.bin',metadata=dict(target='index-tts2.5',kind='native_input'),tensors=new)))
seq=[25,*range(1,10),25,201,25];results=[]
for label,binary in [('baseline',baseline),('candidate',candidate)]:
 lines=''.join(f'flow\t{root}/input-{s}\t{root}/{label}-{i}.bin\n' for i,s in enumerate(seq))
 p=subprocess.run([str(Path(binary).resolve()),'--serve','bundles/full'],input=lines,text=True,capture_output=True,check=True)
 rows=[json.loads(s) for s in p.stdout.splitlines()];assert rows[0]['ready'] and len(rows)==len(seq)+1
 assert all(r.get('ok') if steps!=201 else not r.get('ok') for steps,r in zip(seq,rows[1:]))
 results.append(rows[1:])
comparisons=[]
for i,steps in enumerate(seq):
 if steps==201:continue
 a,b=results[0][i],results[1][i];assert a['shape']==b['shape'];values=[]
 for label in ['baseline','candidate']:
  v=array.array('f');v.frombytes((root/f'{label}-{i}.bin').read_bytes());values.append(v)
 assert len(values[0])==len(values[1]);errors=[x-y for x,y in zip(*values)];peak=max(map(abs,errors));rmse=math.sqrt(sum(x*x for x in errors)/len(errors));comparisons.append(dict(steps=steps,max_abs=peak,rmse=rmse,passed=peak<1e-4 and rmse<1e-5))
for label in ['baseline','candidate']:
 assert (root/f'{label}-0.bin').read_bytes()==(root/f'{label}-10.bin').read_bytes()==(root/f'{label}-12.bin').read_bytes()
report=dict(passed=all(r['passed'] for r in comparisons),error_recovery=True,cache_eviction_stable=True,results=comparisons)
(root/'comparison.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2));sys.exit(0 if report['passed'] else 1)
