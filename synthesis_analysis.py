import matplotlib.pyplot as plt
import cPickle as pickle
import numpy as np

exps = [17,18,19]
exp_no = exps[0]
t = []
Hs = []
Ks = []
Cs = []
fname = 'res/exp_%d_stats.bin'%exp_no
stats = pickle.load(open(fname,'r'))
Hlim = [stats['tar']['H'], stats['src']['H']]
Klim = [stats['tar']['Kurtosis'], stats['src']['Kurtosis']]
Clim = [stats['tar']['MeanCoh'], stats['src']['MeanCoh']]
for i, exp_no in enumerate(exps):

    fname = 'res/exp_%d_stats.bin'%exp_no
    stats = pickle.load(open(fname,'r'))

    alphas = np.mean(stats['alphas'])
    t.append(alphas)
    Hs.append(stats['syn']['H'])
    Ks.append(stats['syn']['Kurtosis'])
    Cs.append(stats['syn']['MeanCoh'])

feats = [ Hs, Ks, Cs ]
featlims = [Hlim, Klim, Clim]
feat_str = ['H','Kurtosis','Coherence']
for i,(f,l,s) in enumerate(zip(feats,featlims,feat_str)):
    plt.subplot(3,1,i+1), plt.hold(True)
    plt.plot(t,f,'.-'), plt.title(s)
    plt.xlabel(r'$\alpha$')
    plt.plot([0,1],[ l[0], l[0] ],'r')
    plt.plot([0,1],[ l[1], l[1] ],'g')

plt.show()
