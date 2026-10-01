"""Paired fixed-ASD position sweep spanning quiet to severe noise scenarios.
Values are effective additive per-axis broadband ASD, not universal site floors.
"""
from pathlib import Path
import json
import numpy as np
from threadpoolctl import threadpool_limits
from forward_model import QuadrotorConfig,simulate_array,noise_sigma_from_asd,DEFAULT_FS
from localization import extract_fundamental_phasors,estimate_position
from simulation_utils import save_json,provenance
from study4_bearing import summarize
D=Path('simulation_results/v5_noise_sweep')

def main():
 D.mkdir(parents=True,exist_ok=True); asds=[.01,.03,.1,.3,1.,3.];cfg=QuadrotorConfig();nodes=np.array([[0,0,0],[5,0,0],[5,5,0],[0,5,0]]);rng=np.random.default_rng(42); grid=np.linspace(.5,4.5,8);records={n:[] for n in asds};truth=[];attitudes=[];checks=[]
 old=np.load('simulation_results/v2/study4_asd_0.0633822.npz'); index=0
 for ix,x in enumerate(grid):
  for y in grid:
   for _ in range(8):
    pos=np.array([x,y,rng.uniform(.5,1.5)]);att=np.r_[rng.uniform(-np.pi/6,np.pi/6,2),rng.uniform(-np.pi,np.pi)];B=simulate_array(cfg,pos,att,nodes,rng=rng);N=rng.normal(size=B.shape)
    assert np.max(abs(pos-old['truth'][index]))<1e-14;assert np.max(abs(att-old['attitude'][index]))<1e-14
    Y0=extract_fundamental_phasors(B,DEFAULT_FS,cfg);Yn=extract_fundamental_phasors(N,DEFAULT_FS,cfg)
    if index<5:
     sigma=noise_sigma_from_asd(.06338219172543927,DEFAULT_FS);direct=extract_fundamental_phasors(B+sigma*N,DEFAULT_FS,cfg);err=np.linalg.norm(direct-(Y0+sigma*Yn))/np.linalg.norm(direct);assert err<1e-12;checks.append(err)
    for asd in asds:
     Y=Y0+noise_sigma_from_asd(asd,DEFAULT_FS)*Yn;est,info=estimate_position(Y,att,nodes,cfg);records[asd].append((est,info))
    truth.append(pos);attitudes.append(att);index+=1
  print('Noise sweep row',ix+1,flush=True)
 truth=np.array(truth);s={}
 for asd in asds:
  rr=records[asd];est=np.array([r[0] for r in rr]);boundary=np.array([r[1]['at_boundary'] for r in rr]);success=np.array([r[1]['success'] for r in rr]);residual=np.array([r[1]['residual'] for r in rr]);err2=np.linalg.norm(est[:,:2]-truth[:,:2],axis=1);err3=np.linalg.norm(est-truth,axis=1)
  s[str(asd)]={'horizontal_2d':summarize(err2,8,8,11),'position_3d':summarize(err3,8,8,10),'boundary_fraction':float(boundary.mean()),'optimizer_success_fraction':float(success.mean())}
  np.savez_compressed(D/('asd_'+str(asd)+'.npz'),truth=truth,attitude=attitudes,nodes=nodes,grid=grid,asd_pT_sqrtHz=asd,estimate=est,errors_2d=err2,errors_3d=err3,boundary=boundary,success=success,residual=residual)
 save_json(D/'summary.json',s);save_json(D/'checks.json',{'all_512_poses_match_v2':True,'max_linear_extraction_relative_error':max(checks)})
 provenance(D,'noise_sweep',dict(asds_pT_sqrtHz=asds,noise='effective additive independent white per-axis ASD; not a site-specific environmental PSD',seed=42,paired_with='all v2 study4 poses and source/noise realizations; extraction linearity checked',model='original four-rotor source including PWM; known attitude/frequencies/moment ratio; point sensors',literature_context='Cohen 2010 natural ELF/VLF 1-100 fT/sqrtHz; Zhou 2024 emphasizes site-dependent contemporary industrial excess; scenarios are sensitivity values, not literature percentiles'))
 print(json.dumps(s,indent=2))
if __name__=='__main__':
 with threadpool_limits(limits=1):main()
