import argparse,array,json,random,subprocess,wave
from pathlib import Path
parser=argparse.ArgumentParser(description='Compare FFT features on PCM16 audio rates, stereo, silence and noise');parser.add_argument('source',help='PCM16 reference WAV');parser.add_argument('--baseline',default='outputs/perf/dsp-frontend-base');parser.add_argument('--candidate',default='build/itts25-frontend');parser.add_argument('--output',default='outputs/perf/frontend-variants');args=parser.parse_args()
root=Path(args.output);root.mkdir(parents=True,exist_ok=True)
with wave.open(args.source,'rb') as w:
 assert w.getsampwidth()==2
 v=array.array('h',w.readframes(w.getnframes()));sr=w.getframerate();channels=w.getnchannels();v=v[::channels]
variants=[]
for rate in [8000,32000,44100]:
 data=array.array('h');frames=len(v)*rate//sr
 for i in range(frames):
  x=v[min(len(v)-1,i*sr//rate)];data.extend([x,x])
 variants.append((f'stereo-{rate}',rate,2,data))
variants.append(('silence',16000,1,array.array('h',[0]*4800)))
r=random.Random(20261003);variants.append(('noise',22050,1,array.array('h',[r.randrange(-4000,4001) for _ in range(11025)])))
for name,rate,ch,data in variants:
 with wave.open(str(root/f'{name}.wav'),'wb') as w:w.setnchannels(ch);w.setsampwidth(2);w.setframerate(rate);w.writeframes(data.tobytes())
results=[]
for name,rate,ch,data in variants:
 for label,binary in [('baseline',args.baseline),('candidate',args.candidate)]:
  subprocess.run([binary,'--prepare','bundles/frontend',str(root/f'{name}.wav'),'ZH','你好世界',str(root/f'{name}-{label}')],stdout=subprocess.DEVNULL,check=True)
 p=subprocess.run(['python3','tools/compare_input_bundles.py',str(root/f'{name}-baseline'),str(root/f'{name}-candidate')],capture_output=True,text=True,check=True)
 results.append(dict(case=name,result=json.loads(p.stdout)))
result=dict(passed=all(r['result']['passed'] for r in results),results=results);(root/'comparison.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
