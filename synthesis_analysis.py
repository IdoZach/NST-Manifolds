import matplotlib.pyplot as plt
import cPickle as pickle
import numpy as np
from patch_stats import get_stats
#from scipy.misc import imfilter
from sys import argv
from scipy.signal import convolve2d
from scipy.signal import medfilt2d

def gauss2D_filter(shape=(9,9),sigma=0.5,one_D=True):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    if one_D:
        h = np.sum(h,axis=0) # turn to 1D
    h = np.array(h,np.float32)
    return h

filt = gauss2D_filter(shape=(3,3),one_D=False)

if len(argv)>1:
    exps = range(int(argv[1]),int(argv[2])+1)
    print 'using exps (cmd line)', exps
else:
    exps = [17,18,19]
    print 'using exps', exps
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
    im = stats['res_im']
    im = convolve2d(im,filt,mode='same',boundary='symm')
    #plt.imshow(im,cmap='gray')
    #plt.show()

    stats['syn'] = get_stats(im)

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
    #print l
    plt.plot(t,f,'.-'), plt.title('%s,%1.2f...%1.2f'%(s,l[0],l[1]))
    plt.xlabel(r'$\alpha$')
#    plt.plot([0,1],[ l[0], l[0] ],'r')
#    plt.plot([0,1],[ l[1], l[1] ],'g')


plt.show()
