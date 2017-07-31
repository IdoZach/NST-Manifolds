import numpy as np
from fbm2d import hurst2d
from scipy.stats import kurtosis
from coherence import coherence
from fbm_data import synth2

def get_stats(im):
    def standardize(x):
        x=np.array(x)
        x[np.isnan(x)]=0.0
        x=x-np.min(x)
        x=x/np.max(x)
        return x

    patch = standardize(im)
    h,_ = hurst2d(patch,max_tau=7)
    # estimate Gaussianity via kurtosis
    kurt = kurtosis(patch.flatten())
    coh = coherence(patch)
    return dict({'H':h,'Kurtosis':kurt,'MeanCoh':np.mean(coh['logcoh'])})
