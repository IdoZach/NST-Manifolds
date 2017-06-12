import dtcwt
import matplotlib.pyplot as plt
import numpy as np
import cPickle as pickle
from scipy.misc import imresize
from sklearn.mixture import GaussianMixture

import fbm_data
from fbm_data import generate_2d_fbms, get_kth_imgs

class LocalPhase():
    def __init__(self, x_train0, x_test0, y_train0, y_test0, thres=0.2, recalc_stats = False):
        self.thres = thres
        self.train = x_train0
        fname = 'localphase.bin'
        fname2 = 'localphase_y.bin'
        try:
            with open(fname,'r') as f:
                self.model = pickle.load(f)
            print 'loaded saved model'
        except:
            print 'preparing new model'
            self.model = self.get_local_phase_model()
            with open(fname,'w') as f:
                pickle.dump(self.model,f)
            print 'saved file', fname

        # now we have the model, we can estimate weights for each sample in the train/test
        #train_stats = self.get_weight_stats(x_train0)
        try:
            with open(fname2,'r') as f:
                self.agg_y_test, self.agg_y_train = pickle.load(f)
            print 'loaded saved local phase weights'
        except:
            print 'preparing phase weights'
            self.agg_y_test = self.get_weight_stats(x_test0,y_test0)
            self.agg_y_train = self.get_weight_stats(x_train0,y_train0)
            with open(fname2,'w') as f:
                pickle.dump([self.agg_y_test,self.agg_y_train],f)
            print 'saved phase weights'

        if recalc_stats:
            print 'recalculating TEST phase weights'
            self.agg_y_test = self.get_weight_stats(x_test0,y_test0)
            #self.agg_y_train = self.get_weight_stats(x_train0,y_train0)
            with open(fname2,'w') as f:
                pickle.dump([self.agg_y_test,self.agg_y_train],f)
            print 'saved phase weights'
        #print test_stats





    def get_weight_stats(self, dat, y):
        #print dat.shape
        all_stats=[]
        for k in range(dat.shape[0]):
            print 'processing %d%%'%int(100.0*k/dat.shape[0])
            im = dat[k,:,:]
            angs = self.get_vecs(im)

            #print angs.shape
            we = []
            for o in range(6):
                we.append([])

            for i in range(angs.shape[0]):
                for o in range(6):
                    pred =self.model[o].predict_proba([angs[i,:,o]])
                    we[o].append(pred)
            stats = []
            for w in we:
                if len(w)==0:
                    continue
                w1 = np.squeeze(np.array(w),axis=1)
                #wm = np.mean(w1,axis=0)
                wm = np.max(w1,axis=0)
                #print 'w',np.array(w).shape
                #print 'wm',wm.shape
                #ws = np.std(w1,axis=0)
                #stats.append( np.hstack([wm, ws]) )
                #print 'wm',w1
                stats.append(wm)
            if len(stats)>0:
                stats = np.vstack(stats)
                #print 'stats',stats.shape
                #stats = stats[0,:] # take one orientation...

                stats = np.std(stats,axis=0)
                #print 'GOOD',stats
            else:
                stats = np.zeros([angs.shape[1]])
                #print 'BAD-',stats

            all_stats.append(stats)
        #print len(all_stats)
        test_stats=[]
        for i,t in enumerate(all_stats):
            test_stats.append(np.hstack([y[i],t]))
        return np.vstack(test_stats)




    def get_vecs(self,im):
        thres = self.thres
        good_sz = 2**np.round(np.log2(im.shape))
        im = imresize(im,[ int(x) for x in good_sz ])

        #plt.figure(0,figsize=(2,2))
        #plt.imshow(im,interpolation='none')
        #plt.show()

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
                            if ii>=0 and ii<len(Ihp) and xx>=0 and xx<Ihp[ii].shape[0] and yy>=0 and yy<Ihp[ii].shape[1]:
                                vec.append(Ihp[ii][xx,yy,:])
                            #else:
                            #    vec.append([])
                    if len(vec)==10: # full neighborhood
                        vecs.append(vec)

            # now we have the vectors of the elements where each item contains 10 complex entries for each orientation
        mags = []
        angs = []
        #print vecs
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


    def get_local_phase_model(self):
        imgs = self.train
        all_angs = []
        for i,im in enumerate(imgs):
            print '%f%%'%int(i*100.0/len(imgs))
            all_angs.append(self.get_vecs(im))

        print [x.shape for x in all_angs]
        all_angs = np.vstack(all_angs)
        print 'shape',all_angs.shape
        gmms = [ GaussianMixture(n_components=10,covariance_type='diag') for o in range(6) ]

        weights = []
        for o, gmm in enumerate(gmms):
            gmm.fit(all_angs[:,:,o])
        #    weights.append(gmm.weights_)
        #weights = np.array(weights)
        #vars = np.var(weights,axis=0)
        #print vars
        #plt.plot(vars)
        #plt.imshow(np.array(weights),interpolation='none')
        return gmms

##
if __name__=='__main__':
    #im = x_train0[10]
    n=32
    original_dim=784
    x_train0, x_test0, y_train0, y_test0 = \
        get_kth_imgs(N=50000,n=n,reCalc=False,resize=original_dim)
##

    localphase = LocalPhase(x_train0,x_test0,y_train0,y_test0, recalc_stats=True)

