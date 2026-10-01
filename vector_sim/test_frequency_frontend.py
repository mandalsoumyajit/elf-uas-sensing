import unittest,json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from frequency_frontend import preprocess,detect_tones,fit_phasors,FS_RAW
from study6_diagnostics import resolved

class FrequencyChecks(unittest.TestCase):
 def test_independent_four_tone_recovery(self):
  t=np.arange(25000)/FS_RAW; f=np.array([518.3,603.2,727.8,882.4]); g=np.array([[1,.3+.2j,.2j],[.4j,1,.5],[.3,.4j,1],[1,.8j,.3]])
  b=np.real(g.T@np.exp(2j*np.pi*f[:,None]*t)); est,info=detect_tones(preprocess(b))
  self.assertTrue(info['accepted']);self.assertLess(np.max(abs(info['frequencies']-f)),.01)
  self.assertLess(np.max(abs(resolved(est)-resolved(g))),1e-3)
 def test_fewer_than_four_tones_abstains(self):
  t=np.arange(25000)/FS_RAW;b=np.array([np.cos(2*np.pi*610*t),np.cos(2*np.pi*700*t),np.cos(2*np.pi*610*t)])
  self.assertFalse(detect_tones(preprocess(b))[1]['accepted'])
 def test_zero_record_abstains(self):self.assertFalse(detect_tones(preprocess(np.zeros((3,25000))))[1]['accepted'])
 def test_known_phasor_solution(self):
  f=np.array([510,630,750,890]);t=np.arange(1000)/5000; g=np.array([[1,.3j,.2],[.4,1j,.2],[1,.2,.4j],[.5j,1,.7]])
  b=np.real(g.T@np.exp(2j*np.pi*f[:,None]*t))+np.array([.1,.2,.3])[:,None]
  gh,_,_=fit_phasors(b,f);self.assertLess(np.max(abs(gh-g)),1e-12)

if __name__=='__main__':
 with threadpool_limits(limits=1):unittest.main()
