import dtcwt
import matplotlib.pyplot as plt
import numpy as np
from scipy.misc import imresize
from sklearn.mixture import GaussianMixture

import fbm_data
from fbm_data import generate_2d_fbms, get_kth_imgs
x_train0, x_test0, y_train0, y_test0 = \
        get_kth_imgs(N=50000,n=n,reCalc=False,resize=original_dim)
##
def get_local_phase_model(imgs,thres=0.2):
    def get_vecs(im,thres=0.2):
        good_sz = 2**np.round(np.log2(im.shape))
        im = imresize(im,[ int(x) for x in good_sz ])

        plt.figure(0,figsize=(2,2))
        plt.imshow(im,interpolation='none')
        plt.show()

        T = dtcwt.Transform2d()
        Im = T.forward(im,nlevels=3)

        M = Im.highpasses[0].shape[1] # max size
        block = np.zeros([M,M,3,6])
        vecs = []
        Ihp = Im.highpasses
        for i, hp in enumerate(Ihp):
            for x in range(hp.shape[0]):
                for y in range(hp.shape[1]):
                    cur_i = [ [i,x,y] ]
                    neighs = [ [i,x+xx,y+yy] for xx,yy in zip ([1,1,-1,-1],[1,-1,1,-1])]
                    sons = [ [i-1,2*x+xx,2*y+yy] for xx,yy in zip([1,0,1,0],[0,0,1,1]) ]
                    parent = [ [i+1,x/2,y/2] ]

                    all = [cur_i, neighs, sons, parent]
                    vec = []
                    for j in all:
                        for k in j:
                            #print 'k',k
                            ii,xx,yy = k
                            if __name__ == '__main__':
                                if ii>=0 and ii<len(Ihp) and xx>=0 and xx<Ihp[ii].shape[0] and yy>=0 and yy<Ihp[ii].shape[1]:
                                    vec.append(Ihp[ii][xx,yy,:])
                                #else:
                                #    vec.append([])
                    if len(vec)==10: # full neighborhood
                        vecs.append(vec)

            # now we have the vectors of the elements where each item contains 10 complex entries for each orientation
        mags = []
        angs = []

        for v in vecs:
            mags.append( [ np.abs(x) for x in v ])
            angs.append( [ np.angle(x) for x in v ])

        mags = np.array(mags)
        angs = np.array(angs)
        # fit gmm
        mags0 =np.reshape(mags[:,0,:],[-1,1])
        smags = np.sort(mags0,axis=0)
        #print smags
        thressed = smags[int(smags.shape[0]*(1.0-thres))]
        #print thres
        angs = angs[np.mean(mags[:,0,:],axis=1)>thressed,:,:]
        #print angs.shape
    #plt.hist(mags0)
    #plt.gca().invert_yaxis()
    #plt.show()
    #print mags0
        return angs
    all_angs = []
    for i,im in enumerate(imgs):
        print '%f%%'%int(i*100.0/len(imgs))
        all_angs.append(get_vecs(im,thres))

    print [x.shape for x in all_angs]
    all_angs = np.vstack(all_angs)
    print 'shape',all_angs.shape
    gmms = [ GaussianMixture(n_components=10,covariance_type='diag') for o in range(6) ]

    weights = []
    for o, gmm in enumerate(gmms):
        gmm.fit(all_angs[:,:,o])
        weights.append(gmm.weights_)
    weights = np.array(weights)
    vars = np.var(weights,axis=0)
    print vars
    #plt.plot(vars)
    #plt.imshow(np.array(weights),interpolation='none')
    return gmms

if __name__=='__main__':
    #im = x_train0[10]
    gmms = get_local_phase_model(x_train0,thres=0.3)
