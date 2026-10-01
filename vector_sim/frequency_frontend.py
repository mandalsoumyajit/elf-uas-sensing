"""Measured-data-only frequency front end for synthetic records; no truth inputs.
Assumes four stationary tones in a declared 450--950 Hz search interval.
Frequency rank is the feature ordering; this does not identify physical motors.
"""
import numpy as np
from scipy.signal import butter,sosfilt,find_peaks
from scipy.optimize import least_squares

FS_RAW=100000.; FS=5000.; ACQUISITION=.25; WINDOW=.2
SOS=butter(8,1200,fs=FS_RAW,output='sos')

def preprocess(raw):
    # Causal filter with 50 ms startup allowance; 20:1 downsampling.
    return sosfilt(SOS,raw,axis=-1)[:,5000::20]

def fit_phasors(b,freqs):
    t=np.arange(b.shape[-1])/FS
    basis=np.column_stack([np.ones(len(t))]+[v for f in freqs for v in (np.cos(2*np.pi*f*t),np.sin(2*np.pi*f*t))])
    coef=np.linalg.lstsq(basis,b.T,rcond=None)[0]
    g=coef[1::2]-1j*coef[2::2]
    return g,basis@coef-b.T,float(np.linalg.cond(basis))

def detect_tones(b):
    n=b.shape[-1]; nfft=8192
    z=np.fft.rfft((b-b.mean(axis=1,keepdims=True))*np.hanning(n),n=nfft,axis=-1)
    power=np.sum(abs(z)**2,axis=0); freqs=np.fft.rfftfreq(nfft,1/FS)
    band=(freqs>=450)&(freqs<=950); p=power[band]; f=freqs[band]
    # Conservative peak separation of one native 1/T bin. Thresholds fixed.
    floor=np.median(p); peaks,_=find_peaks(p,distance=int(np.ceil((1/WINDOW)/(FS/nfft))),height=max(8*floor,.01*p.max()),prominence=4*floor)
    if len(peaks)<4:return None,{'accepted':False,'peak_count':len(peaks),'reason':'fewer_than_four_peaks'}
    peaks=peaks[np.argsort(p[peaks])[-4:]]; init=np.sort(f[peaks])
    scale=max(np.linalg.norm(b),np.finfo(float).tiny); bs=b/scale
    # Peak-derived local bounds, never truth-derived initialization or bounds.
    lo=np.maximum(init-3.0,450); hi=np.minimum(init+3.0,950)
    def residual(fr):return fit_phasors(bs,fr)[1].ravel()
    result=least_squares(residual,init,bounds=(lo,hi),max_nfev=30,ftol=1e-7,xtol=1e-7,gtol=1e-7)
    order=np.argsort(result.x); est=result.x[order]
    g,res,condition=fit_phasors(b,est)
    accepted=bool(result.success and condition<1000 and np.min(np.diff(est))>1e-3)
    return (g if accepted else None),{'accepted':accepted,'peak_count':len(peaks),'reason':'accepted' if accepted else 'fit_or_condition','frequencies':est,'residual_fraction':float(np.linalg.norm(res)/scale),'condition':condition,'nfev':result.nfev}
