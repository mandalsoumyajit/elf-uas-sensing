"""Test whether independent noise-augmented training improves the operating sweep."""
from pathlib import Path
import numpy as np,json
from threadpoolctl import threadpool_limits
from study6_diagnostics import draw,resolved,CFG
from frequency_frontend import preprocess,FS_RAW
from simulation_utils import make_model,save_json,provenance
D=Path('simulation_results/v4_operating'); ORDER=np.argsort(CFG.motor_freqs_hz*CFG.pole_pairs);F=(CFG.motor_freqs_hz*CFG.pole_pairs)[ORDER]

def main():
 t=np.arange(25000)/FS_RAW; wave=np.exp(2j*np.pi*F[:,None]*(t-.15)); tt=np.arange(1000)/5000
 basis=np.column_stack([np.ones(1000)]+[v for f in F for v in (np.cos(2*np.pi*f*tt),np.sin(2*np.pi*f*tt))]);P=np.linalg.pinv(basis)
 cases=json.loads((D/'frequency_summary.json').read_text());pred={};oracle={}
 for repeat in range(10):
  g,y,_,_=draw(2000,28000+repeat);g=g[:,ORDER,:];x=np.array([resolved(a) for a in g]);rng=np.random.default_rng(43000+repeat)
  for j in range(1000):
   signal=np.real(g[j].T@wave);snr=rng.uniform(5,30);sigma=np.sqrt(np.mean(signal[:,5000:]**2)*FS_RAW/(2*100)/10**(snr/10));raw=signal+sigma*rng.normal(size=signal.shape);b=preprocess(raw);coef=P@b.T;phasor=coef[1::2]-1j*coef[2::2];x[j]=resolved(phasor)
  model=make_model('mlp',repeat).fit(x,y)
  for case in cases:
   a=np.load(D/(case+'.npz'));ok=a['accepted'];p=np.full(len(ok),y.mean())
   if ok.any():p[ok]=model.predict(a['features'][ok])
   pred.setdefault(case,[]).append(p);oracle.setdefault(case,[]).append(model.predict(a['oracle_features']))
  print('Noise-augmented training',repeat+1,flush=True)
 summary={}; rng=np.random.default_rng(44001)
 for case in cases:
  a=np.load(D/(case+'.npz'));y=a['y'];p=np.array(pred[case]);op=np.array(oracle[case]);base=a['predictions'];ok=a['accepted']
  rmse=lambda z,yy:float(np.rad2deg(np.sqrt(np.mean((z-yy)**2))))
  dif=[];vals=[]
  for _ in range(2000):
   rr=rng.integers(10,size=10);ii=rng.integers(len(y),size=len(y));z=p[rr][:,ii];b=base[rr][:,ii];dif.append(rmse(b,y[ii])-rmse(z,y[ii])); vals.append(rmse(z,y[ii]))
  summary[case]={'coverage':float(ok.mean()),'fallback_overall_rmse_deg':rmse(p,y),'accepted_rmse_deg':rmse(p[:,ok],y[ok]) if ok.any() else None,'supplied_frequency_rmse_deg':rmse(op,y),'fallback_rmse_ci95_deg':np.quantile(vals,[.025,.975]).tolist(),'advantage_vs_noiseless_training_ci95_deg':np.quantile(dif,[.025,.975]).tolist()}
  np.savez_compressed(D/('augmented_'+case+'.npz'),predictions=p,oracle_predictions=op,y=y)
 save_json(D/'augmented_summary.json',summary)
 provenance(D,'noise_training',dict(train_seeds=list(range(28000,28010)),noise_seeds=list(range(43000,43010)),total_training_samples=2000,noisy_training_samples=1000,noisy_training_SNR_uniform_dB=[5,30],remaining_samples='noiseless',training_frequencies='nominal and supplied to coefficient fit; no test features or labels used',independent_test='same fixed operating cases, 180 poses',CI='2000 paired resamples of training repeat and test observation; includes mean fallback'))
 print(json.dumps(summary,indent=2))
if __name__=='__main__':
 with threadpool_limits(limits=1):main()
