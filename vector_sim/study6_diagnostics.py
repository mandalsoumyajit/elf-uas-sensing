"""Controlled noiseless identifiability and receiver-mismatch diagnostics.
Fundamental-only model: no claim of measured performance or tone extraction.
Aperture is three co-centred ideal 598 mm square loops, NOT extracted PCB turns.
"""
from pathlib import Path
import json, shutil, hashlib
import numpy as np
from numpy.polynomial.legendre import leggauss
from threadpoolctl import threadpool_limits
from forward_model import QuadrotorConfig, tilt_magnitude, simulate_B_timeseries
from localization import unit_phasors
from features import bandpass, covariance_features
from simulation_utils import make_model, save_json, provenance

OUT=Path('simulation_results/v3_diagnostics')
CFG=QuadrotorConfig(pwm_depth=0)
FS=100000.; DUR=.2
T=np.arange(int(FS*DUR))/FS
F=CFG.motor_freqs_hz*CFG.pole_pairs
D=np.array([v for f in F for v in (np.cos(2*np.pi*f*T), np.sin(2*np.pi*f*T))])
DF=bandpass(D,FS,CFG.feature_center_hz()-50,CFG.feature_center_hz()+50)
GRAM=DF@DF.T/len(T)


def compact(g):
    a=np.empty((3,8)); a[:,0::2]=g.real.T; a[:,1::2]=-g.imag.T
    a=a/np.max(abs(a)); c=a@GRAM@a.T; p=np.diag(c)
    return np.r_[p/p.sum(),[c[i,j]/np.sqrt(p[i]*p[j]) for i,j in [(0,1),(0,2),(1,2)]]]


def resolved(g):
    h=g[:,:,None]*g[:,None,:].conj()
    h=h/np.trace(h,axis1=1,axis2=2).real[:,None,None]
    return np.concatenate([np.r_[np.diag(k).real,[k[i,j].real for i,j in [(0,1),(0,2),(1,2)]],[k[i,j].imag for i,j in [(0,1),(0,2),(1,2)]]] for k in h])


def aperture(position,attitude,side=.598,order=12):
    u,w=leggauss(order); u=u*side/2
    v=np.array(np.meshgrid(u,u,indexing='ij')).reshape(2,-1).T
    weights=np.outer(w,w).ravel()/4
    g=np.empty((4,3),complex)
    for axis in range(3):
        nodes=np.zeros((len(v),3)); nodes[:,[i for i in range(3) if i!=axis]]=v
        g[:,axis]=unit_phasors(position,attitude,nodes,CFG)[:,:,axis]@weights
    return g


def draw(n,seed,enriched=False):
    rng=np.random.default_rng(seed)
    rp=rng.uniform(-np.pi/4,np.pi/4,(n,2))
    # Keep broad-domain support; replace half by samples from +/- 10 deg.
    if enriched: rp[:n//2]=rng.uniform(-np.pi/18,np.pi/18,(n//2,2))
    yaw=rng.uniform(-np.pi,np.pi,n); r=rng.uniform(1,4,n); az=rng.uniform(-np.pi/12,np.pi/12,n)
    meta=np.column_stack([rp,yaw,r*np.cos(az),r*np.sin(az),np.full(n,.3)])
    phase=rng.uniform(0,2*np.pi,(n,4))
    g=np.array([unit_phasors(m[3:],m[:3],[[0,0,0]],CFG)[:,0,:]*np.exp(1j*p[:,None]) for m,p in zip(meta,phase)])
    return g,tilt_magnitude(rp[:,0],rp[:,1]),meta,phase


def features(g): return {'compact':np.array([compact(a) for a in g]),'resolved':np.array([resolved(a) for a in g])}


def metrics(y,p):
    e=np.rad2deg(np.asarray(p)-y); mask=y<=np.deg2rad(5)
    return {'rmse_deg':float(np.sqrt(np.mean(e**2))),'bias_deg':float(np.mean(e)), 'p95_abs_deg':float(np.quantile(abs(e),.95)), 'near_zero_rmse_deg':float(np.sqrt(np.mean(e[...,mask]**2)))}


def physical_checks():
    pos=np.array([1.4,.2,.3]); att=np.array([.1,-.2,.6]); g=unit_phasors(pos,att,[[0,0,0]],CFG)[:,0,:]
    _,b=simulate_B_timeseries(CFG,pos,*att,FS,DUR,phase_offsets=np.zeros(4))
    reconstructed=np.real(g.T@np.exp(2j*np.pi*F[:,None]*T))
    err=np.linalg.norm(reconstructed-b)/np.linalg.norm(b)
    assert err<1e-12
    ferr=np.max(abs(compact(g)-covariance_features(bandpass(b,FS,CFG.feature_center_hz()-50,CFG.feature_center_hz()+50))))
    assert ferr<1e-11
    tiny=aperture(pos,att,side=1e-5)
    terr=np.linalg.norm(tiny-g)/np.linalg.norm(g); assert terr<1e-9
    assert np.max(abs(resolved(g)-resolved(g*np.exp(1j*np.arange(4))[:,None])))<1e-12
    # Integral reproduces a constant field: normalized Gauss weights sum to one.
    assert abs(np.outer(leggauss(12)[1],leggauss(12)[1]).sum()/4-1)<1e-14
    return dict(time_phasor_relative_error=err,compact_feature_max_error=ferr,vanishing_aperture_relative_error=terr)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    checks=physical_checks(); print('Physical checks',checks,flush=True)
    gtest,y,meta,phase=draw(1200,27001)
    # Additional independent zero-tilt records; kept separate from domain metric.
    gz,yz,mz,pz=draw(160,27002); mz[:,:2]=0; yz[:]=0
    gz=np.array([unit_phasors(m[3:],m[:3],[[0,0,0]],CFG)[:,0,:]*np.exp(1j*p[:,None]) for m,p in zip(mz,pz)])
    xt=features(gtest); xz=features(gz)
    perturb={}
    for gain in [.01,.05,.10]: perturb[f'gain_pm{int(gain*100)}pct']=gtest*np.array([1-gain,1,1+gain])
    for deg in [1,5,10]:
        perturb[f'phase_pm{deg}deg']=gtest*np.exp(1j*np.deg2rad([-deg,0,deg]))
        a=np.deg2rad(deg); m=np.array([[np.cos(a),np.sin(a),0],[0,np.cos(a),np.sin(a)],[np.sin(a),0,np.cos(a)]])
        perturb[f'axis_{deg}deg']=gtest@m.T
    perturb['square_598mm']=np.array([aperture(m[3:],m[:3])*np.exp(1j*p[:,None]) for m,p in zip(meta,phase)])
    xp={k:features(v) for k,v in perturb.items()}
    pred={}; zeropred={}; mismatch={}
    for repeat in range(10):
        for enriched in [False,True]:
            gtr,ytr,_,_=draw(2000,28000+repeat,enriched); xtr=features(gtr)
            for feat in (['compact'] if enriched else ['compact','resolved']):
                for modelkind in ['mlp','rf']:
                    key=f'{modelkind}_{feat}_'+('enriched' if enriched else 'broad')
                    model=make_model(modelkind,repeat).fit(xtr[feat],ytr)
                    pred.setdefault(key,[]).append(model.predict(xt[feat]))
                    zeropred.setdefault(key,[]).append(model.predict(xz[feat]))
                    if not enriched:
                        for case,xx in xp.items(): mismatch.setdefault(key+'__'+case,[]).append(model.predict(xx[feat]))
        print(f'Training repeat {repeat+1}/10',flush=True)
    summary={}
    for k,p in pred.items():
        row=metrics(y,p); z=np.rad2deg(zeropred[k]); row['zero_rmse_deg']=float(np.sqrt(np.mean(z*z))); row['zero_bias_deg']=float(np.mean(z)); row['repeat_rmse_deg']=[metrics(y,a)['rmse_deg'] for a in p]; summary[k]=row
    ms={k:metrics(y,p) for k,p in mismatch.items()}
    np.savez_compressed(OUT/'inclination_predictions.npz',y=y,meta=meta,zero_meta=mz,**{k:np.array(v) for k,v in pred.items()},**{'zero__'+k:np.array(v) for k,v in zeropred.items()},**{'mismatch__'+k:np.array(v) for k,v in mismatch.items()})
    save_json(OUT/'inclination_summary.json',summary); save_json(OUT/'mismatch_summary.json',ms)
    # Independent paired geometry sweep, identical attitude/azimuth/phases at each range.
    _,_,poses,ph=draw(128,29001); ranges=np.array([.5,.75,1,1.5,2,3,4,6,10]); errors=[]; ratios=[]; conv=[]
    for radius in ranges:
        er=[]; ra=[]; co=[]
        for m,p in zip(poses,ph):
            pos=m[3:].copy(); pos[:2]*=radius/np.linalg.norm(pos[:2])
            gp=unit_phasors(pos,m[:3],[[0,0,0]],CFG)[:,0,:]
            ga=aperture(pos,m[:3],order=16); gb=aperture(pos,m[:3],order=24)
            er.append(np.linalg.norm(gb-gp)/np.linalg.norm(gp))
            co.append(np.linalg.norm(gb-ga)/np.linalg.norm(gb))
            ep=np.exp(1j*p[:,None]); ra.append(np.max(abs(compact(gb*ep)[:3]-compact(gp*ep)[:3])))
        errors.append(er); ratios.append(ra); conv.append(co)
    np.savez_compressed(OUT/'aperture_sweep.npz',ranges=ranges,relative_field_error=errors,power_ratio_departure=ratios,quadrature_relative_error=conv)
    aps={str(r):{'median_relative_field_error':float(np.median(e)),'p95_relative_field_error':float(np.quantile(e,.95)),'max_ratio_departure':float(np.max(v)),'max_quadrature_error':float(np.max(c))} for r,e,v,c in zip(ranges,errors,ratios,conv)}
    save_json(OUT/'aperture_summary.json',aps)
    checks['max_16_vs_24_quadrature_relative_error']=float(np.max(conv)); assert np.max(conv)<1e-5
    # Paired repeat-level uncertainty: test set held fixed, training-repeat variance only.
    rng=np.random.default_rng(31001); paired={}
    for kind in ['mlp','rf']:
        baseline=np.asarray(pred[kind+'_compact_broad']); bz=np.asarray(zeropred[kind+'_compact_broad'])
        for alt in ['compact_enriched','resolved_broad']:
            k=kind+'_'+alt; p=np.asarray(pred[k]); z=np.asarray(zeropred[k]); a=[]; b=[]
            for _ in range(4000):
                ix=rng.integers(10,size=10)
                a.append(float(np.rad2deg(np.sqrt(np.mean((baseline[ix]-y)**2))-np.sqrt(np.mean((p[ix]-y)**2)))))
                b.append(float(np.rad2deg(np.sqrt(np.mean(bz[ix]**2))-np.sqrt(np.mean(z[ix]**2)))))
            paired[k]={'domain_rmse_advantage_ci95_deg':np.quantile(a,[.025,.975]).tolist(),'zero_rmse_advantage_ci95_deg':np.quantile(b,[.025,.975]).tolist()}
    save_json(OUT/'paired_training_intervals.json',paired); save_json(OUT/'physical_checks.json',checks)
    provenance(OUT,'study6',dict(training_samples=2000,training_repeats=10,test_samples=1200,zero_test_samples=160,model='noiseless four distinct fundamental phasors; ideal known tone separation',compact='exact finite-record filtered covariance via Gram matrix',resolved='36 real normalized per-tone Hermitian entries',enrichment='half broad +/-45 deg and half +/-10 deg roll/pitch, same total count',aperture='three co-centred ideal .598 m square surfaces; no PCB turn extraction',quadrature='tensor Gauss-Legendre 16 vs 24; orientation uses 12',calibration='deterministic fixed uncorrected per-axis perturbations; hypothetical tolerances',interval='paired resampling of 10 training repeats; fixed test ensemble, not physical confidence bounds'))
    print(json.dumps(summary,indent=2)); print('Aperture',json.dumps(aps,indent=2))

if __name__=='__main__':
    with threadpool_limits(limits=1): main()
