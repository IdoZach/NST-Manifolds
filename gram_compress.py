from keras.models import Sequential, Model, Input, Layer
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D
from keras.layers import Lambda
from keras.layers import merge
from keras.optimizers import SGD, RMSprop, Adam
from keras import backend as K
from numpy.linalg import norm
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from os.path import exists
from fbm_data import synth2
import cPickle as pickle
import cv2, numpy as np
import numpy as np
from numpy.linalg import svd, matrix_rank
from scipy.linalg import svd as ssvd
from scipy.sparse.linalg import svds
from sklearn.decomposition import PCA
##
"""
G0 = pickle.load(open('G0.bin','r'))
##
G = G0#[:-1]
gg = []
gg_trans = []
# 10 components work fine
PCA_10 = PCA(n_components=10)
PCA_3 = PCA(n_components=5)
plt.figure()
for i, g in enumerate(G):
    if i<=2:
        PCA_ = PCA_10
    else:
        PCA_ = PCA_3
    res = PCA_.fit(g)
    transformed = PCA_.transform(g)
    gg_trans.append(transformed)
    compressed = PCA_.inverse_transform(transformed)
    print compressed.shape
    gg.append(compressed)
    plt.semilogy(res.explained_variance_ratio_[:10])

    print g.shape,res.explained_variance_ratio_[:10]
plt.legend(range(len(G)))
plt.show()
pickle.dump(gg,open('G0_comp.bin','w'))
"""
##
#plt.imshow(gg_trans[0],interpolation='none')

# todo separator build autoencoder to compress G matrices
##
#all_G = pickle.load(open('KTH_G.bin','r'))
##

def save_svds(all_F,lengths=[2,2,4,4,4],k=20):
    for ii in range(10):
        print 'im',ii
        F = all_F[ii]
        UU=[]
        ss=[]
        VV=[]
        SS=[]
        for f, ll in zip(F,lengths):
            chunk = f.shape[0]/ll
            lU=[]
            ls=[]
            lV=[]
            lS=[]
            for l in range(ll):
                cur = f[chunk*l:chunk*(l+1),:]
                print 'calc svd im %d inner %d'%(ii,l),
                #U,s,V = svd(f)
                #print f.shape
                U,s,V = svds(cur,k=k) # sprase svd with 20 largest singular values
                if np.any(s==0): # for some reason if there are absolute zeros they come up after the highest sing val.
                    zero_locations= np.where(s==0)[0]
                    print 'U before',U.shape, V.shape
                    nz_locations = range(len(s))[:int(zero_locations[0])]
                    s = np.concatenate([s[zero_locations],s[nz_locations]])
                    U = np.concatenate([U[:,zero_locations],U[:,nz_locations]],axis=1)
                    V = np.concatenate([V[zero_locations,:],V[nz_locations,:]],axis=0)
                    print 'U after',U.shape, V.shape

                # in svds s are *increasing*
                #U,s,V = ssvd(f)
                print 'done'
                lU.append(U)
                lV.append(V)
                ls.append(s)
                S = np.zeros([U.shape[0],V.shape[0]])
                S[:s.shape[0],:s.shape[0]]=np.diag(s)
                lS.append(S)
            UU.append(lU)
            VV.append(lV)
            ss.append(ls)
            SS.append(lS)
        print 'saving...',
        pickle.dump([UU,ss,VV,SS],open('svd_res_%d.bin'%ii,'w'))
        print 'done'
        # each UU,ss,VV,SS contains a list, each list is the response between max pools, and within
        # we find another list of the inner convolution layers.

all_F = pickle.load(open('KTH_F.bin','r'))
##
save_svds(all_F)
##
# todo separator modifiction of singular values
##
#all_G = pickle.load(open('KTH_G.bin','r'))
##
ii=0
UU,ss,VV,SS = pickle.load(open('svd_res_%d.bin'%ii,'r'))
##
lengths = [2,2,4,4,4] # number of response matrices within each F
def to_flat(LL):
    res_all = []
    for L in LL:
        res = []
        for U in L:
            cur = []
            for u in U:
                cur.append(u)
            res+=cur
        res_all.append(res)
    return res_all
def to_tree(LL,lengths):
    res_all = []
    for U in LL:
        k=0
        cur = []
        for l in lengths:
            cur.append(U[k:k+l])
            k+=l
        res_all.append(cur)
    return res_all

ss_mat = np.array(to_flat([ss])[0]).T
#
#ss_mat = np.array(ss).T
plt.show()
plt.hold(False)
def fact_i(lengths):
    x=[]
    for i,l in enumerate(lengths):
        x+=[i for k in range(l)]
    return np.array(x)
fact = np.array([2.0**-fact_i(lengths)])
ll1 = np.log(ss_mat*fact)
for j in range(ss_mat.shape[1]):
    plt.plot(np.log(ss_mat[:,j]*fact[:,j]))
    plt.show(block=False)
    plt.hold(True)
    print j
    plt.pause(0.5)

# todo separator
##
ii=9
UU,ss,VV,SS = pickle.load(open('svd_res_%d.bin'%ii,'r'))
##
plt.hold(True)
lengths = [2,2,4,4,4] # number of response matrices within each F

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y = np.concatenate([ np.ones(box_pts/2)*y[0], y, np.ones(box_pts/2)*y[-1]] )
    y_smooth = np.convolve(y, box, mode='same')
    y_smooth = y_smooth[box_pts/2:-box_pts/2+1]

    return y_smooth

plt.clf()
plt.figure(1)
plt.hold(False)
def processF(img_i,load_i=None,save_i=None,
             load_singular=False,load_mean=True,load_std=True,
             load_levels=range(16)):
    new_G=[]
    do_load = load_i is not None
    do_save = not do_load
    UU,ss,VV,SS = pickle.load(open('svd_res_%d.bin'%img_i,'r'))
    if do_load:
        desc = pickle.load(open('desc_%d.bin'%load_i,'r'))
    else: # save
        desc = {'s':[],'mean':[],'std':[]}

    cur_lev=0
    for l,(lU,ls,lV,lS) in enumerate(zip(UU,ss,VV,SS)): # level 0,1,2,3,4
        f_lev=[]
        for il,(tU,ts,tV,tS) in enumerate(zip(lU,ls,lV,lS)): # lengths 2 2 4 4 4 4

            # process inner s
            #ss1 = np.log(ss1) +np.random.randn(*ss1.shape)*0.2
            #print 'inner', tV.shape, tU.shape
            if True:#all(ts>0) :#and False:
                new_s = np.log(ts)
                if do_save:
                    desc['s'].append(new_s)
                else:
                    if load_singular and cur_lev in load_levels:
                        new_s=desc['s'][0]
                    del desc['s'][0]
                new_s = np.exp(new_s)
            else:
                new_s = ts.copy()
            #ss1 = ss1 + np.random.randn(*ss1.shape)*0.01#(1-alpha)*ss1+alpha*new_s[:,j]

            #new_s[-1] = ts[-1].copy()*(1+np.random.randn()*0.5)
            new_s[new_s<0.0]=1e-5
            S_new = np.diag(new_s)
            inner1=np.dot(S_new,tV)
            inner_f = np.dot(tU,inner1) # this is g1
            orig_f = np.dot(tU, np.dot( np.diag(ts),tV))
            # keep original mean and std
            use_mean = np.mean(orig_f,axis=0)
            use_std = np.std(orig_f,axis=0)
            if do_load:
                if load_mean and cur_lev in load_levels:
                    use_mean = desc['mean'][0]
                if load_std and cur_lev in load_levels:
                    use_std = desc['std'][0]
                del desc['mean'][0]
                del desc['std'][0]
            elif do_save:
                desc['mean'].append(use_mean)
                desc['std'].append(use_std)

            inner_f-=np.mean(inner_f,axis=0)
            inner_f = inner_f / np.std(inner_f,axis=0) * use_std + use_mean
            inner_f[np.isnan(inner_f)]=0.0
            f_lev.append(inner_f)

            cur_lev+=1

            plt.semilogy(ts)
            plt.hold(True)
            plt.semilogy(new_s,'--')

        F_lev = np.concatenate(f_lev,axis=0)
        print F_lev.shape
        G_lev = np.dot(F_lev.T,F_lev)
        print G_lev.shape
        new_G.append(G_lev)

    if do_save:
        pickle.dump(desc,open('desc_%d.bin'%save_i,'w'))
    if do_load:
        desc_out = pickle.load(open('desc_%d.bin'%load_i,'r'))
    pickle.dump(new_G,open('new_G.bin','w'))

    return new_G, desc_out
##
new_G, desc = processF(9,load_i=2,load_std=False,
                       load_levels=range(0,14))

# now we see that we can use the mean and std vectors only

#do_legend(lengths)
##
for ii in range(10):
    print 'processing',ii
    processF(ii,save_i=ii)
##


