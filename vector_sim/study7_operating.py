"""Frequency robustness, deployment failure summaries, and full-stage CPU timing."""
from pathlib import Path
import time,json,platform
import numpy as np
from threadpoolctl import threadpool_limits
from study6_diagnostics import draw,resolved,CFG
from frequency_frontend import preprocess,detect_tones,fit_phasors,FS_RAW,FS,ACQUISITION,WINDOW
from simulation_utils import make_model,save_json,provenance
from localization import estimate_position,unit_phasors

OUT=Path('simulation_results/v4_operating')
ORDER=np.argsort(CFG.motor_freqs_hz*CFG.pole_pairs)
F=(CFG.motor_freqs_hz*CFG.pole_pairs)[ORDER]

def features(gs): return np.array([resolved(g) for g in gs])
def rmse(y,p):return float(np.rad2deg(np.sqrt(np.mean((p-y)**2))))
def scenario_records(g,case,seed):
    rng=np.random.default_rng(seed); t=np.arange(int(FS_RAW*ACQUISITION))/FS_RAW
    freqs=F*case.get('speed',1)
    if 'gapscale' in case:freqs=F.mean()+(F-F.mean())*case['gapscale']
    drift=case.get('drift',0)*np.array([1,-1,1,-1])/WINDOW
    phase=2*np.pi*(freqs[:,None]*(t-.15)+.5*np.asarray(drift)[:,None]*(t-.15)**2)
    exp=np.exp(1j*phase)
    data=[]; oracle=[]; accepted=[]; estimated=[]; times=[]; residuals=[]; timings_raw=[]
    for k,gi in enumerate(g):
        signal=np.real(gi.T@exp)
        # Reference SNR: mean signal power / expected white noise in 100 Hz.
        sigma=np.sqrt(np.mean(signal[:,5000:]**2)*FS_RAW/(2*100)/10**(case['snr']/10))
        noise=rng.normal(size=signal.shape)
        if case.get('correlated',False):
            common=rng.normal(size=(1,signal.shape[1])); noise=np.sqrt(.5)*noise+np.sqrt(.5)*common
        raw=signal+sigma*noise
        t0=time.perf_counter(); b=preprocess(raw); t1=time.perf_counter(); gd,info=detect_tones(b); t2=time.perf_counter()
        gf=resolved(gd) if gd is not None else np.zeros(36); t3=time.perf_counter()
        go,_,_=fit_phasors(b,freqs)
        data.append(gf); oracle.append(resolved(go)); accepted.append(info['accepted']); estimated.append(info.get('frequencies',np.full(4,np.nan))); residuals.append(info.get('residual_fraction',np.nan)); times.append([t1-t0,t2-t1,t3-t2]); timings_raw.append(raw if k<3 else None)
    return np.array(data),np.array(oracle),np.array(accepted),np.array(estimated),np.array(times),np.array(residuals),freqs


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    models=[]; mean=[]
    for i in range(10):
        g,y,_,_=draw(2000,28000+i); x=features(g[:,ORDER,:]); model=make_model('mlp',i).fit(x,y); models.append(model); mean.append(y.mean())
    g,y,meta,phase=draw(180,37001); g=g[:,ORDER,:]
    cases={
        'nominal_minus5dB':dict(snr=-5),'nominal_10dB':dict(snr=10),'nominal_20dB':dict(snr=20),'nominal_30dB':dict(snr=30),
        'speed80_20dB':dict(snr=20,speed=.8),'speed120_20dB':dict(snr=20,speed=1.2),
        'gap4p375Hz_20dB':dict(snr=20,gapscale=.25),'gap1p75Hz_20dB':dict(snr=20,gapscale=.1),
        'drift5Hz_20dB':dict(snr=20,drift=5),'drift20Hz_20dB':dict(snr=20,drift=20),
        'correlated50pct_20dB':dict(snr=20,correlated=True)}
    summary={}; all_times=[]
    for name,case in cases.items():
        x,xo,ok,fhat,times,res,frequencies=scenario_records(g,case,38001)
        p=[]; po=[]; inference=[]
        for model,m in zip(models,mean):
            pp=np.full(len(y),m)
            if ok.any():pp[ok]=model.predict(x[ok])
            p.append(pp);po.append(model.predict(xo))
        p=np.array(p);po=np.array(po)
        # Single-record MLP latency, after warmup, independent of batched scoring.
        for xx in xo:
            t0=time.perf_counter();models[0].predict(xx[None,:]);inference.append(time.perf_counter()-t0)
        timing=np.column_stack([times,inference]); all_times.append(timing)
        errorfreq=np.max(abs(fhat-frequencies),axis=1)
        row={'coverage':float(ok.mean()),'frequency_all_within_1Hz_fraction':float(np.mean(ok & (errorfreq<=1))), 'accepted_rmse_deg':rmse(y[ok],p[:,ok]) if ok.any() else None,'fallback_overall_rmse_deg':rmse(y,p),'supplied_frequency_rmse_deg':rmse(y,po),'mean_fallback_rmse_deg':rmse(y,np.array(mean)[:,None]),'accepted_p95_abs_deg':float(np.quantile(abs(np.rad2deg(p[:,ok]-y[ok])),.95)) if ok.any() else None,'frequency_max_error_median_Hz':float(np.nanmedian(errorfreq)) if np.isfinite(errorfreq).any() else None,'processing_median_ms':float(np.median(timing.sum(1))*1e3),'processing_p95_ms':float(np.quantile(timing.sum(1),.95)*1e3)}
        # Paired repeat and observation resampling; includes mean fallback on rejected records.
        rng=np.random.default_rng(39001); delta=[]; vals=[]
        for _ in range(2000):
            rr=rng.integers(10,size=10); ii=rng.integers(len(y),size=len(y))
            yy=y[ii]; pp=p[rr][:,ii]; oo=po[rr][:,ii]
            delta.append(rmse(yy,pp)-rmse(yy,oo)); vals.append(rmse(yy,pp))
        row['fallback_rmse_ci95_deg']=np.quantile(vals,[.025,.975]).tolist(); row['excess_vs_supplied_frequency_ci95_deg']=np.quantile(delta,[.025,.975]).tolist()
        np.savez_compressed(OUT/(name+'.npz'),y=y,meta=meta,accepted=ok,estimated_frequencies=fhat,true_center_frequencies=frequencies,features=x,oracle_features=xo,predictions=p,oracle_predictions=po,timing_seconds=timing,residual=res)
        summary[name]=row; print(name,json.dumps(row),flush=True); save_json(OUT/'frequency_summary.json',summary)
    # Independent grid-centre baseline and failure audit of all retained v2 fits.
    failure={}
    for name in ['0.0633822','10']:
        a=np.load('simulation_results/v2/study4_asd_'+name+'.npz'); e=a['errors_2d']; center=np.linalg.norm(a['truth'][:,:2]-[2.5,2.5],axis=1); b=a['boundary']; success=a['success']; bad=e>.5
        rng=np.random.default_rng(40001); differences=[]
        for _ in range(4000):
            ix=(np.arange(64)[:,None]*8+rng.integers(8,size=(64,8))).ravel();differences.append(np.sqrt(np.mean(e[ix]**2))-np.sqrt(np.mean(center[ix]**2)))
        failure[name]={'estimator_horizontal_rmse_m':float(np.sqrt(np.mean(e*e))),'array_center_rmse_m':float(np.sqrt(np.mean(center*center))),'excess_vs_center_ci95_m':np.quantile(differences,[.025,.975]).tolist(),'error_over_half_meter_fraction':float(bad.mean()),'boundary_fraction':float(b.mean()),'boundary_precision_for_half_meter_error':float(bad[b].mean()) if b.any() else None,'boundary_recall_for_half_meter_error':float(b[bad].mean()) if bad.any() else None,'nonboundary_bad_fraction':float(bad[~b].mean()),'success_bad_fraction':float(bad[success].mean()),'p95_error_m':float(np.quantile(e,.95))}
    save_json(OUT/'position_failure_summary.json',failure)
    # Additional conditional-position compute budget: tone extraction with supplied
    # frequencies plus nonlinear inversion, on exactly specified synthetic records.
    nodes=np.array([[0,0,0],[5,0,0],[5,5,0],[0,5,0]])
    rng=np.random.default_rng(41001); stages=[]
    for _ in range(40):
        pos=rng.uniform([.5,.5,.5],[4.5,4.5,1.5]);att=rng.uniform([-.5,-.5,-np.pi],[.5,.5,np.pi]); gp=unit_phasors(pos,att,nodes,CFG)[ORDER]*np.exp(1j*rng.uniform(0,2*np.pi,(4,1,1)))
        t=np.arange(25000)/FS_RAW; raw=np.real(np.einsum('kna,kt->nat',gp,np.exp(2j*np.pi*F[:,None]*t))); raw+=rng.normal(0,.0633822e-12*np.sqrt(FS_RAW/2),raw.shape)
        t0=time.perf_counter(); filtered=np.array([preprocess(b) for b in raw]); t1=time.perf_counter(); ph=np.array([fit_phasors(b,F)[0] for b in filtered]).transpose(1,0,2);t2=time.perf_counter();estimate,info=estimate_position(ph[np.argsort(ORDER)],att,nodes,CFG);t3=time.perf_counter();stages.append([t1-t0,t2-t1,t3-t2])
    a=np.concatenate(all_times);stages=np.array(stages)
    timing={'cpu':'Intel Core Ultra 7 265K','logical_processors':20,'BLAS_threads':1,'raw_fs_Hz':FS_RAW,'analysis_fs_Hz':FS,'raw_samples_per_axis':25000,'analysis_samples_per_axis':1000,'acquisition_s':ACQUISITION,'retained_window_s':WINDOW,'startup_allowance_s':.05,'orientation_stage_names':['causal_filter_decimation','frequency_detection_and_fit','features','single_MLP_prediction'],'orientation_stage_median_ms':(np.median(a,axis=0)*1e3).tolist(),'orientation_total_median_ms':float(np.median(a.sum(1))*1e3),'orientation_total_p95_ms':float(np.quantile(a.sum(1),.95)*1e3),'position_stage_names':['causal_filter_decimation','supplied_frequency_fit','nonlinear_position_fit'],'position_stage_median_ms':(np.median(stages,axis=0)*1e3).tolist(),'position_total_median_ms':float(np.median(stages.sum(1))*1e3),'position_total_p95_ms':float(np.quantile(stages.sum(1),.95)*1e3),'raw_float64_buffer_bytes_three_axes':3*25000*8,'raw_float64_buffer_bytes_four_vector_nodes':12*25000*8,'resolved_MLP_parameters':4481,'resolved_MLP_multiply_accumulate_count':4384,'measurement_limits':'desktop wall-clock warmed Python; excludes ADC/I/O and scheduling; position frequencies and attitude still supplied; no dropped-record or embedded test'}
    save_json(OUT/'timing_summary.json',timing);np.savez_compressed(OUT/'timing_records.npz',orientation=a,position=stages)
    provenance(OUT,'study7',dict(cases=cases,test_poses=180,training_repeats=10,training_examples=2000,train_seeds=list(range(28000,28010)),test_seed=37001,noise_seed=38001,search_Hz=[450,950],tone_count=4,peak_gate='height max(8*median,0.01*maximum), prominence 4*median, separation >=5Hz',frequency_fit='peak-derived +/-3Hz bounds; 30 function evaluations; no truth initialization',noise='expected white noise in 100Hz relative to mean clean signal power; correlated case equal per-axis variance with off-diagonal coefficient 0.5',drift='alternating +/- linear frequency ramps, stated change over 0.2s; all physical poses stationary',association='frequency rank only; no physical motor association solved',fallback='training-mean inclination on rejected records; all-record RMSE includes these',model='fundamental-only source, nominal point receiver, causal common channel filter; trained on noiseless features',statistics='2000 paired two-way resamples of training repetitions and test observations; conditional on synthetic model'))
    print('Failure',failure);print('Timing',timing)

if __name__=='__main__':
    with threadpool_limits(limits=1):main()
