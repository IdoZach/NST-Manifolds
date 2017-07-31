# https://gist.github.com/baraldilorenzo/8d096f48a1be4a2d660d

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
from os import remove
from glob import glob
from fbm_data import synth2, get_kth_imgs, get_other_imgs
from fbm2d import hurst2d
from scipy.stats import kurtosis
from coherence import coherence
from patch_stats import get_stats
import cPickle as pickle
import cv2, numpy as np
from sklearn.decomposition import PCA
from numpy.linalg import svd
from scipy.sparse.linalg import svds
#from mu_analysis import MuAnalysis

from mpl_toolkits.mplot3d import Axes3D
from sklearn import linear_model

class mu_AE():
    def __init__(self, encoding_dim=20):
        self.sz  =[ 64,  64, 128, 128, 256, 256, 256, 256, 512, 512, 512, 512, 512,
                    512, 512, 512]
        self.inputs=[]
        sz_intermediate_layer  =[ 64 for i in range(len(self.sz))]
        self.enc_inter=[]
        for lsz,s in zip(sz_intermediate_layer,self.sz):
            self.inputs.append(Input(shape=(s,) ))
            self.enc_inter.append(Dense(lsz,activation='tanh'))
        sz_join_layers = [1024, encoding_dim]
        activations = ['tanh','tanh']
        self.enc_join_layers =[]
        for act, sz in zip(activations,sz_join_layers):
            self.enc_join_layers.append(Dense(sz,activation=act))

        # dec
        self.dec_join_layers = []
        for sz in sz_join_layers[::-1]:
            self.dec_join_layers.append(Dense(sz,activation='tanh'))
        self.dec_out = []
        for lsz,s in zip(sz_intermediate_layer,self.sz):
            self.dec_out.append(Dense(s,activation='tanh'))

        # build network
        _tomerge=[]
        for input,inter in zip(self.inputs,self.enc_inter):
            _tomerge.append(inter(input))
        encoded = merge(_tomerge,mode='concat',concat_axis=1)
        #print encoded
        for l in self.enc_join_layers:
            encoded = l(encoded)

        def create_decoder(encoded):
            outputs=[]
            decoded = Lambda(lambda x: x)(encoded)
            for l in self.dec_join_layers:
                decoded = l(decoded)
            for l in self.dec_out:
                outputs.append(l(decoded))
            return outputs
        self.outputs = create_decoder(encoded)


        self.ae = Model(inputs=self.inputs,outputs=self.outputs)

        decoder_input = Input(shape=(encoding_dim,))
        self.decoder = Model(inputs=decoder_input,outputs=create_decoder(decoder_input))
        self.encoder = Model(inputs=self.inputs,outputs=encoded)
        print 'built model'

    def train(self,x_train,x_test,epochs=100):
        self.ae.compile(optimizer='adam', loss='mean_squared_error')
        self.ae.fit(x_train, x_train,
                epochs=epochs,
                batch_size=128,
                shuffle=True,
                validation_data=(x_test, x_test))

class SynData():

    def get_G(self,texture, im):
        #texture = Texture()

        #saved_G = 'KTH_G.bin'
        #saved_F = 'KTH_F.bin'
        #gg = []
        #ff = []

        im = np.stack([im,im,im],axis=2).astype(np.float32) # grayscale
        _, F0 = texture.synTexture(im,onlyGram = True)
        return F0

    def get_svds(self,F,lengths=[2,2,4,4,4],k=20):
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
                #print 'calc svd inner %d'%l,
                #U,s,V = svd(f)
                #print f.shape
                U,s,V = svds(cur,k=k) # sprase svd with 20 largest singular values
                if np.any(s==0): # for some reason if there are absolute zeros they come up after the highest sing val.
                    zero_locations= np.where(s==0)[0]
                    #print 'U before',U.shape, V.shape
                    nz_locations = range(len(s))[:int(zero_locations[0])]
                    s = np.concatenate([s[zero_locations],s[nz_locations]])
                    U = np.concatenate([U[:,zero_locations],U[:,nz_locations]],axis=1)
                    V = np.concatenate([V[zero_locations,:],V[nz_locations,:]],axis=0)
                    #print 'U after',U.shape, V.shape

                # in svds s are *increasing*
                #U,s,V = ssvd(f)
                #print 'done'
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
        #print 'saving...',
        #pickle.dump([UU,ss,VV,SS],open('svd_res_%d.bin'%ii,'w'))
        print 'done svd'
        return UU,ss,VV,SS
        # each UU,ss,VV,SS contains a list, each list is the response between max pools, and within
        # we find another list of the inner convolution layers.

    def processF(self,UsVS,load_i=None,save_i=None,
                 load_singular=False,load_mean=True,load_std=True,
                 load_levels=range(16)):
        new_G=[]
        do_load = load_i is not None
        do_save = not do_load
        UU,ss,VV,SS = UsVS
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
                        #print 'cur s',new_s
                        if load_singular and cur_lev in load_levels:
                            new_s=desc['s'][0]
                        #print 'new s',new_s
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
            #print F_lev.shape
            G_lev = np.dot(F_lev.T,F_lev)
            #print G_lev.shape
            new_G.append(G_lev)

        if do_save:
            pickle.dump(desc,open('desc_%d.bin'%save_i,'w'))
            desc_out = desc
        if do_load:
            desc_out = pickle.load(open('desc_%d.bin'%load_i,'r'))
        pickle.dump(new_G,open('new_G.bin','w'))

        return new_G, desc_out

    def prepare_data(self,idxs=None):
        if self.dataset is 'kth':
            print 'loading from KTH dataset'
            x_train0, x_test0, y_train0, y_test0 = \
                get_kth_imgs(N=50000,n=self.n,reCalc=False,resize=self.original_dim)
            x_train0[0],_ = synth2(224,H=0.3)
            x_train0[1],_ = synth2(224,H=0.5)
            for i in range(2):
                x_train0[i]-=np.min(x_train0[i]*1.0)
                x_train0[i]=x_train0[i]/np.max(x_train0[i])*256.0
            print 'two first samples are fBm'

        else:
            print 'loading from OTHER (SIM) dataset'
            x_train0, x_test0, y_train0, y_test0 = \
                get_other_imgs(N=1000,n=self.n,reCalc=False,resize=self.original_dim)
            #print 'NOT IMPLEMENTED'

        if self.dataset is 'kth' and idxs is not None:
            return x_train0[idxs], x_test0, y_train0[idxs], y_test0
        else:
            return x_train0, x_test0, y_train0, y_test0

    def __init__(self, step_save=False, use_ae=False, dataset='kth'):
        ### here we run with 'save_step=True' to save data.
        ### then we run with 'False' to save the encoded information and G matrices
        ### this can be used in mu_analysis.py to analyse and in vgg_19_keras.py to generate
        print 'creating syn object'
        self.step_save = step_save
        self.use_ae = use_ae
        self.original_dim = 224**2
        self.n=200
        self.weights = np.array([16,16,4,4,2,2,2,2,1,1,1,1,1,1,1,1],dtype=np.float32) # scaling of means according to inverse matrix size
        self.dataset=dataset


    def save_step(self):
        # prepare training data
        x_train0, x_test0, y_train0, y_test0 = self.prepare_data()
        # process
        m_train = []
        m_test = []
        texture = Texture()
        def get_desc(im):
            F0 = self.get_G(texture,im)
            UsVS = self.get_svds(F0)
            _, desc = self.processF(UsVS,save_i=0,load_std=False)
            return desc

        print 'saving results...'

        for i, im in enumerate(x_train0):
            print 'train', i+1, '/', len(x_train0)
            desc=get_desc(im)
            m_train.append(desc)
            pickle.dump(m_train,open('m_train_'+self.dataset+'.bin','w'))

        for i, im in enumerate(x_test0):
            print 'test', i+1, '/', len(x_test0)
            desc=get_desc(im)
            m_test.append(desc)
            pickle.dump(m_test,open('m_test_'+self.dataset+'.bin','w'))

        pickle.dump([y_train0,y_test0],open('m_y_'+self.dataset+'.bin','w'))
        return m_train, m_test, y_train0, y_test0

    def load_step(self):
        print 'loading data and preprocessing train_vars_'+self.dataset+'.bin'
        m_train = pickle.load(open('m_train_'+self.dataset+'.bin','r'))
        m_test = pickle.load(open('m_test_'+self.dataset+'.bin','r'))
        y_train0, y_test0 = pickle.load(open('m_y.bin','r'))

        sel_indexes = []
        if self.dataset is 'kth':
            choose_classes = ['woola','woolb','woolc','woold']
            #choose_classes = []

            use_all = len(choose_classes)==0
            print 'use all classes',use_all
            m_train2 = []
            y_train2 = []
            sel_indexes = []
            for i,(tr,te) in enumerate(zip(m_train,y_train0)):
                if te[2] in choose_classes or use_all:
                    m_train2.append(tr)
                    y_train2.append(te)
                    sel_indexes.append(i)
            pickle.dump(y_train2,open('yy.bin','w'))
            print 'full len',len(m_train),
            m_train = m_train2
            print 'used len',len(m_train)

        train_means = [ m['mean'] for m in m_train ]
        test_means =  [ m['mean'] for m in m_test ]
        train_stds = [ m['std'] for m in m_train ]
        test_stds =  [ m['std'] for m in m_test ]
        train_s = [ m['s'] for m in m_train ]
        test_s =  [ m['s'] for m in m_test ]
        self.m_train = m_train
        self.m_test = m_test

        train_vars = []
        test_vars = []
        train_vars_std = []
        test_vars_std = []
        train_vars_s = []
        test_vars_s = []
        for i in range(len(train_means[0])): # 16
            train_vars.append( np.array([x[i] for x in train_means ]))
            test_vars.append( np.array([x[i] for x in test_means ]))
            train_vars_std.append( np.array([x[i] for x in train_stds ]))
            test_vars_std.append( np.array([x[i] for x in test_stds ]))
            train_vars_s.append( np.array([x[i] for x in train_s ]))
            test_vars_s.append( np.array([x[i] for x in test_s ]))
        # apply weights only to means
        train_vars = [t*w for t,w in zip(train_vars,self.weights)]
        test_vars = [t*w for t,w in zip(test_vars,self.weights)]

        # try pca for the entire dataset
        pickle.dump([train_vars, train_vars_std, train_vars_s],open('train_vars_'+self.dataset+'.bin','w'))
        if self.use_ae:
            AE = mu_AE(encoding_dim=10) # was 10 for one class
            #print len(train_vars[0])
            if exists('ae_model.bin'):# and False:
                print 'loading model from disk...'
                AE.ae.load_weights('ae_model.bin')
            else:
                print 'training model...'
                epochs=1000
                AE.train(train_vars,test_vars,epochs=epochs)
                print 'saving model...'
                AE.ae.save('ae_model.bin')
        self.sel_indexes = sel_indexes
        return train_vars, train_vars_std, train_vars_s

    def save_new_G(self,ii=None,order=None,alphas=None,load=None,exp_no=0):
        print 'saving new G for synthesis'
        x_train0, x_test0, y_train0, y_test0 = self.prepare_data(self.sel_indexes)
        print 'LEN XTRAIN',len(x_train0)
        def distort_latent(all,all_y,one,i_source,distort_dim=[0],distort_amount=[1.01]):
            PCA_ = PCA()
            pca = PCA_.fit(all)

            transformed_source = PCA_.transform(all[i_source])
            transformed_all = PCA_.transform(all)
            transformed = PCA_.transform(one)
            close_point = np.argsort(np.sum(np.square(transformed_all - transformed),axis=1))
            closest_point_ord = 50
            i_source = close_point[closest_point_ord]
            transformed_source = PCA_.transform(all[i_source])
            # distortion by moving to another point
            transformed[0] = transformed_source[0]
            # distortion of manifold parameters by fixed amounts
            #for d,a in zip(distort_dim,distort_amount):
            #    transformed[0][d]*=a # modification

            inverted = PCA_.inverse_transform(transformed)
            inverted = np.array([all[i_source]])
            return inverted, i_source, [PCA_.explained_variance_ratio_, PCA_.explained_variance_]

        def interpolate_latent(all,one,i_source,alpha=0.5):
            closest_point_ord = 3

            PCA_ = PCA()
            pca = PCA_.fit(all)

            transformed_source = PCA_.transform(all[i_source])
            transformed_all = PCA_.transform(all)
            transformed = PCA_.transform(one)
            close_point = np.argsort(np.sum(np.square(transformed_all - transformed),axis=1))

            i_source2 = close_point[closest_point_ord]
            transformed_source = PCA_.transform(all[i_source2])
            # distortion by averaging in latent space
            transformed[0] = (1-alpha)*transformed_source[0]+alpha*transformed[0]

            inverted = PCA_.inverse_transform(transformed)
            #inverted = np.array([all[i_source]]) # override
            return inverted, i_source2, [PCA_.explained_variance_ratio_, PCA_.explained_variance_]


        if self.use_ae:
            print 'predicting all training set'
            all_pred = AE.ae.predict(train_vars)
            all_enc = AE.encoder.predict(train_vars)
            print 'saving in and out of sample',ii
            train_one = [ np.array([t[ii]]) for t in train_vars ]
            desc_mean_out = AE.ae.predict(train_one)
            desc_mean_out = [1.0*t/w for t,w in zip(desc_mean_out,self.weights)]

            # apply distortion on latent dimension
            one_encoded = AE.encoder.predict(train_one)
            #i_source = 15
            print 'enc',one_encoded
            one_encoded, i_source, explained_variance = interpolate_latent(all_enc,one_encoded,
                                                                       i_source=ii,
                                                                       alpha=alpha)
            desc_mean_out_modified = AE.decoder.predict(one_encoded)

        # using pca instead of AE+pca
        load_file = 'train_vars_'+self.dataset+'.bin'
        #load_file = 'train_vars_comp.bin'
        print 'loading from file',load_file
        train_vars_compressed, train_vars_std_compressed, train_vras_s_compressed = pickle.load(open(load_file,'r'))

        def interpolate_latent_pca(vars,i_cur,i_source0,alpha=0.5,closest_point_ord=3):
            inverted=[]
            i_source=None
            for cur in vars:
                #PCA_ = PCA(n_components=20)
                PCA_ = PCA()
                pca = PCA_.fit(cur)

                c_from = PCA_.transform(cur[i_cur].reshape(1,-1))
                if i_source is None: # set only once
                    c_all = PCA_.transform(cur)
                    i_source = np.argsort(np.sum(np.square(c_all - c_from),axis=1))[closest_point_ord]
                c_to = PCA_.transform(cur[i_source].reshape(1,-1))
                #print 'alpha', alpha
                c_from[0] = (1-alpha)*c_to[0]+alpha*c_from[0]
                inv1 = PCA_.inverse_transform(c_from)
                inv2 = cur[i_cur]
                #print 'inv1', inv1
                #print 'inv2', inv2

                inverted.append(inv1)
                #print 'debug'
                #inverted.append(inv2)
            return inverted, i_source#, [PCA_.explained_variance_ratio_, PCA_.explained_variance_]

        if ii is None:
            ii=2
        alpha=0.0
        if order is None:
            order = 3

        i2 = 10
        i_source = i2
        if alphas is None:
            alphas = np.ones(16)*alpha
        #alphas[8:]=1.0
        if load is None:
            load = {'mean':True, 'std': True, 's': True}
        #alphas[:]=0.0

        print 'PARAMS: exp %d ii %d order %d alphas'%(exp_no,ii,order),alphas,'load',load
        with open('res/exp_%d_log.txt'%exp_no,'w') as f:
            strr = 'PARAMS: exp %d ii %d order %d alphas %s load %s'%(exp_no,ii,order,str(alphas),str(load))
            f.write(strr)


        #desc_mean_out_modified = [ np.array([t[ii]]) for t in train_vars_compressed ]
        #desc_mean_out_modified2 = [ np.array([t[i2]]) for t in train_vars_compressed ]
        #desc_mean_out_modified = [ a*g+(1-a)*h for a,g,h in
        #                              zip(alphas,desc_mean_out_modified,desc_mean_out_modified2)]

        desc_std_out = [ np.array([t[ii]]) for t in train_vars_std_compressed ]


        desc_mean_out_modified, i2_chosen = interpolate_latent_pca(train_vars_compressed,ii,i2,alpha,order)

        # TODO why not interpolate in PCA space also the STD???

        i2 = i2_chosen
        #i2 = ii # control
        desc_std_out2 = [ np.array([t[i2]]) for t in train_vars_std_compressed ]
        desc_std_out_modified = [ a*g+(1-a)*h for a,g,h in
                                      zip(alphas,desc_std_out,desc_std_out2)]
        desc_mean_out = [1.0*t/w for t,w in zip(desc_mean_out_modified,self.weights)]

        # s from svd
        texture=Texture()
        UsVS = self.get_svds(self.get_G(texture,x_train0[i2]))

        print 'using modified latent variables'

        # postprocess
        def postprocess(m):
            out=[]
            for mean in m:
                mean[mean<0]=0.0
                out.append(mean)
            return out
        #desc_mean_out = postprocess(desc_mean_out)
        new_desc = self.m_train[ii].copy() # point 1 for interpolation
        #print [ [np.mean(xx), np.std(xx)] for xx in new_desc['mean'] ]
        desc_comparison = {'old':new_desc.copy()}
        new_desc['mean'] = desc_mean_out
        new_desc['std'] = desc_std_out_modified

        # TODO why not interpolate singular values also?

        new_desc['s'] = [np.log(y) for x in UsVS[1] for y in x] # UsVS[1] is the s (singular)
        # interpolate also stds
        mul = lambda x,y,a: [a*g+(1-a)*h for g,h in zip(x,y)]
        #new_desc['std'] = mul(new_desc['std'],source_desc['std'],alpha)
        #source_desc = self.m_train[i_source].copy() # point 2 for interpolation
        #new_desc['mean'] = source_desc['mean']
        #new_desc['std'] = source_desc['std']
        desc_comparison['new'] = new_desc

        plt.hold(False)
        i=0
        if self.use_ae:
            plt.hold(False)
            plt.plot(explained_variance[1])
            plt.pause(1)

        plt.plot(desc_comparison['old']['mean'][i].T)
        plt.hold(True)
        plt.plot(desc_comparison['new']['mean'][i].T,'r--')
        plt.show(block=False)
        plt.pause(1)

        pickle.dump(desc_comparison,open('desc_comp.bin','w'))
        pickle.dump(new_desc,open('desc_%d.bin'%ii,'w'))

        if self.use_ae:
            pickle.dump([all_pred, all_enc, y_train0],open('all_pred.bin','w'))
        print 'saving images...'
        plt.imsave('res/exp_%d_im_src_%d.png'%(exp_no,i2),x_train0[i2]/255.0,cmap=plt.cm.gray)
        plt.imsave('res/exp_%d_im_%d.png'%(exp_no,ii),x_train0[ii]/255.0,cmap=plt.cm.gray)
        stats_src = get_stats(x_train0[i2]/255.0)
        stats_tar = get_stats(x_train0[ii]/255.0)

        F0 = self.get_G(texture,x_train0[ii])
        UsVS = self.get_svds(F0)
        self.new_G, self.desc_out = self.processF(UsVS,load_i=ii,load_mean=load['mean'],load_std=load['std'], load_singular=load['s'])
            # this uses desc file and saves new_G to be used by vgg_19_keras.py file

        return stats_src, stats_tar

    def getParams(self):
        return self.new_G, self.desc_out



def get_gram_error(G0,G0_symb,N_l,M_l):
    costl = 0.0
    weights = np.array([1e9 for i in N_l])
    weights[-1] =weights[-1]*0
    for g,gs,n,m,w in zip(G0,G0_symb,N_l,M_l,weights):
        #costl+= 1.0/4.0/n**2/m**2 * w * K.sum(K.square(g-gs))
        costl+= 1.0/4.0/n**2 * w * K.sum(K.square(g-gs))
    #cost = merge(costl,mode='sum')
    return costl

def get_gram_matrices_symb(model,sel):
    #shapes=[64,64,128,128,256,256,256,256,512,512,512,512,512,512,512,512]
    shapes = [64,128,256,512,512]
    lengths = [2,2,4,4,4]
    imsz = [224,112,56,28,14]
    #imsz=[224,224,112,112,56,56,56,56,28,28,28,28,14,14,14,14]
    M_l = [ i**2 for i in imsz ]
    N_l = [ s*l for s,l in zip(shapes,lengths) ]

    G = []
    outputs = [layer.output for layer in model.layers]
    outputs = [ l for i,l in enumerate(outputs) if i in sel]
    #print 'outs',outputs
    k=0
    for i,s in enumerate(lengths):
        act_mat = []
        for j in range(s):
            a=Lambda(lambda x: x[0])(outputs[k+j])
            act_mat.append(Lambda(lambda x: K.squeeze(K.reshape(x,[1,imsz[i]**2,shapes[i]]),axis=0))(a))

        k+=s
        act_mat1 = merge(act_mat,mode='concat',concat_axis=0)
        g = Lambda(lambda x: K.dot(K.transpose(x),x)/imsz[i]**2)(act_mat1)
        #g = Lambda(lambda x: K.dot(K.transpose(x),x))(act_mat1)
        G.append(g)

    return G, N_l, M_l


def get_gram_matrices(activations):
    G = []
    F = []
    shapes = [64,128,256,512,512]
    lengths = [2,2,4,4,4]
    imsz = [224,112,56,28,14]
    k=0
    for i,s in enumerate(lengths):
        act_mat=[]
        for j in range(s):
            a=activations[k+j][0]
            act_mat.append(np.squeeze(np.reshape(a,[1,a.shape[1]*a.shape[2],a.shape[-1]])))
            #print [xx.shape for xx in act_mat]
            #print act_mat.shape
        act_mat=np.concatenate(act_mat,axis=0)
        k+=s
        #print 'running SVD, mat size ', act_mat.shape
        f = act_mat / imsz[i]
        F.append(f)
        #U,s,V = svd(act_mat)
        #print 'done'
        #print 'REC',np.sum(np.square(U*S*V.T-act_mat))
        g = np.dot(act_mat.transpose(),act_mat)/imsz[i]**2
        # now we have the gram matrix g
        G.append(g)
    return G, F

def get_activations(model, input, layers):
    inp = model.input                                           # input placeholder
    outputs = [layer.output for layer in model.layers]          # all layer outputs
    functors = [K.function([inp], [out]) for out in outputs]  # evaluation functions

    # Testing
    layer_outs = [func([input]) for func in functors]
    layer_outs = [ l for i,l in enumerate(layer_outs) if i in layers]
    #print layer_outs

    return layer_outs

def VGG_19_1(weights_path=None,onlyconv=False,caffe=False):
    model = Sequential()
    model.add(Conv2D(64, (3, 3), name='conv1_1', padding='same',activation='relu', batch_input_shape=(1,224,224,3), trainable=False)) #1
    model.add(Conv2D(64, (3, 3), name='conv1_2', padding='same', activation='relu', trainable=False)) #3
    model.add(AveragePooling2D((2,2), strides=(2,2))) #4

    model.add(Conv2D(128, (3, 3), name='conv2_1', padding='same', activation='relu', trainable=False))
    model.add(Conv2D(128, (3, 3), name='conv2_2', padding='same', activation='relu', trainable=False))
    model.add(AveragePooling2D((2,2), strides=(2,2)))

    model.add(Conv2D(256, (3,3),  name='conv3_1', padding='same', activation='relu', trainable=False))
    model.add(Conv2D(256, (3,3),  name='conv3_2', padding='same', activation='relu', trainable=False))
    model.add(Conv2D(256, (3,3),  name='conv3_3', padding='same', activation='relu', trainable=False))
    model.add(Conv2D(256, (3,3),  name='conv3_4', padding='same', activation='relu', trainable=False))
    model.add(AveragePooling2D((2,2), strides=(2,2)))

    model.add(Conv2D(512, (3,3), name='conv4_1', padding='same',activation='relu', trainable=False))
    model.add(Conv2D(512, (3,3), name='conv4_2', padding='same', activation='relu', trainable=False))
    model.add(Conv2D(512, (3,3), name='conv4_3', padding='same', activation='relu', trainable=False))
    model.add(Conv2D(512, (3,3), name='conv4_4', padding='same', activation='relu', trainable=False))
    model.add(AveragePooling2D((2,2), strides=(2,2)))

    model.add(Conv2D(512, (3,3), name='conv5_1', padding='same', activation='relu', trainable=False))
    model.add(Conv2D(512, (3,3), name='conv5_2', padding='same', activation='relu', trainable=False))
    model.add(Conv2D(512, (3,3), name='conv5_3', padding='same', activation='relu', trainable=False))
    model.add(Conv2D(512, (3,3), name='conv5_4', padding='same', activation='relu', trainable=False))
    model.add(AveragePooling2D((2,2), strides=(2,2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    #model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    #model.add(Dropout(0.5))
    model.add(Dense(1000, activation='softmax'))

    if caffe:
        weights_data = np.load("dataout.h5").item()

        for layer in model.layers:
            if layer.name in weights_data.keys():
                #print 'loading layer',layer.name
                layer_weights = weights_data[layer.name]

                layer.set_weights((layer_weights['weights'],
                    layer_weights['biases']))
    else:
        if weights_path:
            #print model.get_weights()[0][0][0]
            model.load_weights(weights_path)
            #print model.get_weights()[0][0][0]

    if onlyconv:
        for i in range(5): # get rid of fc layers
            #print 'popping', model.layers[-1]
            model.pop()
    return model

def VGG_19(weights_path=None,onlyconv=False):
    model = Sequential()
    model.add(ZeroPadding2D((1,1),batch_input_shape=(1,224,224,3))) #0
    model.add(Conv2D(64, (3, 3), activation='relu')) #1
    model.add(ZeroPadding2D((1,1))) #2
    model.add(Conv2D(64, (3, 3), activation='relu')) #3
    model.add(MaxPooling2D((2,2), strides=(2,2))) #4

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(256, (3,3),  activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(256, (3,3),  activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(256, (3,3),  activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(256, (3,3),  activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3,3),  activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3,3),  activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3,3),  activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3,3),  activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3,3),  activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3,3),  activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3,3),  activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3,3),  activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1000, activation='softmax'))

    if weights_path:
        #print model.get_weights()[0][0][0]
        model.load_weights(weights_path)
        #print model.get_weights()[0][0][0]
    if onlyconv:
        for i in range(7): # get rid of fc layers
            model.pop()
    return model


cur_iter=1
class Texture():
    def __init__(self, use_caffe=True, exp_no=0, stats=None):
        self.model = VGG_19_1('vgg19_weights_tf_dim_ordering_tf_kernels.h5',onlyconv=True,caffe=use_caffe)
        self.use_caffe = use_caffe
        self.exp_no = exp_no
        self.stats = stats
    def synTexture(self, im=None, G0_from=None,onlyGram=False,maxiter=500):
        model = self.model
        use_caffe = self.use_caffe
        conv_layers = [0,1,3,4,6,7,8,9,11,12,13,14,16,17,18,19]
        mm=np.array([ 0.40760392,  0.45795686,  0.48501961])
        if im is not None:
            im0=im

            if not use_caffe:
                im[:,:,0] -= 103.939
                im[:,:,1] -= 116.779
                im[:,:,2] -= 123.68
            else: # caffe
                im = im/255.0

                im[:,:,0]=im[:,:,0]-np.mean(im[:,:,0])+mm[0] # b
                im[:,:,1]=im[:,:,1]-np.mean(im[:,:,1])+mm[1] # g
                im[:,:,2]=im[:,:,2]-np.mean(im[:,:,2])+mm[2] # r


            #im = im.transpose((2,0,1))
            im = np.expand_dims(im, axis=0)
            # Test pretrained model

            #model = VGG_19_1('dataout.h5',onlyconv=True)
            #vgg19_weights.h5
            #sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
            #model.compile(optimizer=sgd, loss='categorical_crossentropy')
            print 'predicting'
            out = model.predict(im)

            #conv_layers = [1,3,6,8,11,13,15,17,20,22,24,26,29,31,33,35]

            activations = get_activations(model,im,conv_layers)

            G0, F0 = get_gram_matrices(activations)
        elif G0_from is not None:
            G0 = pickle.load(open(G0_from,'r'))
            print 'using saved G0 file (probably compressed)'


        #G0 = compress_gram(G0)

        if onlyGram:
            return G0, F0

        G0_symb, N_l, M_l = get_gram_matrices_symb(model,sel=conv_layers)

        errf = get_gram_error(G0,G0_symb,N_l,M_l)
        #print errf
        grads = K.gradients(errf,model.input)
        #opt = Adam()

        #opt.get_gradients
        #updates = opt.get_updates([model.input],[],[errf])
        #train = K.function([model.input],[errf, model.input],updates=updates)
        coef=0.5 if use_caffe else 128.0
        gray_im = np.random.randn(224,224)
        im_iter = np.stack([gray_im,gray_im,gray_im],axis=2)*coef
        #im_iter = np.random.randn(im0.shape[0],im0.shape[1],im0.shape[2])*coef
        for i in range(3):
            if use_caffe:
                im_iter[:,:,i]=im_iter[:,:,i]+mm[i]
            else:
                im_iter[:,:,i]=im_iter[:,:,i]+0*128.0


        iters = 10000
        plt.figure()
        _,pp=plt.subplots(2,2)
        costs=[]
        grads_fun = K.function([model.input],grads)
        cost_fun = K.function([model.input],[errf])

        """
        x = Input(batch_shape=(1,224,224,3))
        #opt_model = Model(inputs=x,outputs=G0_symb)
        class CustomLossLayer(Layer):
            def __init__(self, **kwargs):
                self.is_placeholder = True
                super(CustomLossLayer, self).__init__(**kwargs)

            def call(self, inputs):
                loss = get_gram_error(G0,inputs,N_l,M_l)
                self.add_loss(loss, inputs=inputs)
                return inputs

        y = CustomLossLayer()(G0_symb) # G0_symb is a list
        print y

        opt_model = Model([model.input], y)
        rmsprop = RMSprop()
        opt_model.compile(optimizer='rmsprop', loss=None)
        epochs=100
        opt_model.fit([im_iter],
            shuffle=True,
            epochs=epochs,
            verbose=2,
            batch_size=1)

        res =  opt_model.predict(im_iter)
        plt.imshow(im_iter)
        plt.show(block=False)
        plt.pause(0.01)
        """
        #"""
        if use_caffe:
            ranges = [-1,4]
            limits = [0.0,1.0]
        else:
            ranges = [-8,-3]
            limits =[-255, 255.0]
        def limit_im(im):
            im[im<limits[0]]=limits[0]
            im[im>limits[1]]=limits[1]
            return im
        def best_stepsize(im0, grad, steps=np.logspace(ranges[0],ranges[1],15)):
            best = -1
            best_loss = np.inf
            for s in steps:
                new_im = limit_im(im0-s*grad)
                #loss = cost_fun([np.expand_dims(im0-s*grad,0)])[0]
                loss = cost_fun([np.expand_dims(new_im,0)])[0]
                #print loss
                if loss<best_loss:
                    best = s
                    best_loss = loss
            return best

        # bfgs?
        imsize = [224,224,3]
        #maxiter = 2000
        method = 'L-BFGS-B'
        #method = 'BFGS'
        global cur_iter
        cur_iter=0
        global stats_im
        stats_im = []
        global res_im
        res_im = []
        def callback(x):
            global cur_iter
            global stats_im
            global res_im
            #print 'random number',np.random.rand(1)
            if not cur_iter%15:
                im = np.reshape(x,imsize)[:,:,::-1]
                im = np.sum(im,axis=2)/3.0 # turn to grayscale
                plt.imshow(im,cmap=plt.cm.gray)
                plt.imsave('res/syn_res.png',im,cmap=plt.cm.gray)
                if cur_iter==225:
                    plt.imsave('res/exp_%d_res_%d.png'%(self.exp_no,cur_iter),im,cmap=plt.cm.gray)
                    # saveing all stats
                    stats_im = get_stats(im)
                    stats_src, stats_tar = self.stats
                    res_im = im
                    with open('res/exp_%d_stats.txt'%self.exp_no,'w') as f:
                        f.write('stats src')
                        f.write(str(stats_src))
                        f.write('\nstats tar')
                        f.write(str(stats_tar))
                        f.write('\nstats syn')
                        f.write(str(stats_im))
                    print 'SAVED STATS'
                plt.title('iter %d'%cur_iter)
            cur_iter+=1
            plt.show(block=False)
            plt.pause(0.01)
        m=20
        reshape_im = lambda x: np.reshape(x, imsize)
        bounds = (np.ones_like(im_iter)*limits[0],np.ones_like(im_iter)*limits[1])
        bounds = zip(bounds[0].flatten(),bounds[1].flatten())
        min_fun = lambda x: cost_fun([np.expand_dims(reshape_im(x),0)])[0].astype(np.float64)
        grad_fun = lambda x: np.squeeze(grads_fun([np.expand_dims(reshape_im(x),0)])[0]).flatten().astype(np.float64)
        try:
            res = minimize(min_fun,im_iter.flatten().astype(np.float64),
                       jac=grad_fun,method=method,bounds=bounds,callback=callback,
                       options={'disp': False, 'maxiter':maxiter,
                                'maxcor': m, 'ftol': 0, 'gtol': 0})
        except:
            pass
        stats_src, stats_tar = self.stats
        return stats_src, stats_tar, stats_im, res_im
        #pp[0,0].imshow(im_iter)
        #pp[0,1].imshow(res[::-1])
        #plt.show(block=False)
        #plt.pause(0.01)
        #raw_input('done')
        #"""
        """
    for i in range(iters):
        grads_ = grads_fun([np.expand_dims(im_iter,0)])
        #print grads_[0].shape
        grad = np.squeeze(grads_[0])
        #errf1, new_img = train([im_iter])
        costs.append(cost_fun([np.expand_dims(im_iter,0)])[0])
        s = best_stepsize(im_iter,grad)
        #print 'stepsize',s
        im_iter = limit_im(im_iter - s*grad)

        if i%10==0:

            print i, 'norm grad',norm(err), 'cost',costs[-1], 'stepsize', s
            pp[0,1].hold(False)
            pp[0,1].semilogy(costs)
            pp[0,0].imshow(im_iter)
            plt.show(block=False)
            plt.pause(0.01)
    #"""

def get_G_kth():
    texture = Texture()
    original_dim = 224**2
    n=200
    x_train0, x_test0, y_train0, y_test0 = \
        get_kth_imgs(N=50000,n=n,reCalc=False,resize=original_dim)
    saved_G = '/home/ido/disk/KTH_G.bin'
    saved_F = '/home/ido/disk/KTH_F.bin'
    gg = []
    ff = []
    x_train0 = x_train0
    x_train0[0],_ = synth2(224,H=0.3)
    x_train0[1],_ = synth2(224,H=0.5)
    for i in range(2):
        x_train0[i]-=np.min(x_train0[i]*1.0)
        x_train0[i]=x_train0[i]/np.max(x_train0[i])*256.0
    print 'two first samples are fBm'
    #plt.imshow(x_train0[0])

    for i,im in enumerate(x_train0):
        print i, '/', len(x_train0)
        im = np.stack([im,im,im],axis=2).astype(np.float32) # grayscale
        G0, F0 = texture.synTexture(im,onlyGram = True)
        gg.append(G0)
        ff.append(F0)
    pickle.dump(gg,open(saved_G,'w'))
    pickle.dump(ff,open(saved_F,'w'))


def getIm(fbm=False):
    if not fbm:
        kth_path1 = '/home/ido/combined_model_classification/KTH-TIPS2-b/wool/sample_a/'
        images=['pebbles.jpg','cat.jpg']
        kth_imgs = ['22a-scale_3_im_3_col.png', '22a-scale_9_im_9_col.png']
        sel_img = images[1]
        sel_img = kth_path1 + kth_imgs[1]
        kth_path2 = '/home/ido/combined_model_classification/KTH-TIPS2-b/cracker/sample_b/'
        sel_img = kth_path2 + '60b-scale_6_im_10_col.png'
        im = cv2.resize(cv2.imread(sel_img), (224, 224)).astype(np.float32)
        plt.imshow(im)
    else:
        fbm1, _ = synth2(N=224,H=0.2)

        fbm1 = np.stack([fbm1,fbm1,fbm1],axis=2)
        fbm1 = fbm1-np.min(fbm1)
        fbm1 = fbm1/np.max(fbm1)*256.0


        im = fbm1
        plt.imshow(im/256.0)
    plt.show(block=False)
    plt.pause(1)
    return im
if __name__ == "__main__":
    # generate F,G from a set of images
    #get_G_kth()
    #print 'saved F,G'
    exp_no = 0 # ii = 2
    exps = []
    # alpha 1 keeps ii's data

    # 0..5
    exps.append({'ii':3,'order':150,'alphas':np.ones(16)*1.0,'load': {'mean':False,'std':False,'s':False} })
    exps.append({'ii':3,'order':150,'alphas':np.ones(16)*0.0,'load': {'mean':True,'std':True,'s':True} })
    exps.append({'ii':5,'order':10,'alphas':np.ones(16)*1.0,'load': {'mean':False,'std':False,'s':False} })
    exps.append({'ii':5,'order':10,'alphas':np.ones(16)*0.0,'load': {'mean':True,'std':True,'s':True} })
    exps.append({'ii':10,'order':30,'alphas':np.ones(16)*1.0,'load': {'mean':False,'std':False,'s':False} })
    exps.append({'ii':10,'order':30,'alphas':np.ones(16)*0.0,'load': {'mean':True,'std':True,'s':True} })

    # 6..8
    exps.append({'ii':8,'order':50,'alphas':np.ones(16)*0.8,'load': {'mean':True,'std':True,'s':True} })
    exps.append({'ii':10,'order':50,'alphas':np.ones(16)*0.8,'load': {'mean':True,'std':True,'s':True} })
    exps.append({'ii':15,'order':50,'alphas':np.ones(16)*0.8,'load': {'mean':True,'std':True,'s':True} })

    # 9..12
    exps.append({'ii':20,'order':50,'alphas':np.ones(16)*0.8,'load': {'mean':True,'std':True,'s':True} })
    exps.append({'ii':30,'order':40,'alphas':np.ones(16)*0.8,'load': {'mean':True,'std':True,'s':True} })
    alphas = np.ones(16)
    alphas[6:] = 1.0
    exps.append({'ii':40,'order':10,'alphas':alphas,'load': {'mean':True,'std':True,'s':True} })
    exps.append({'ii':50,'order':10,'alphas':alphas,'load': {'mean':True,'std':True,'s':True} })

    # 13..16
    exps.append({'ii':60,'order':30,'alphas':alphas,'load': {'mean':True,'std':True,'s':True} })
    exps.append({'ii':61,'order':30,'alphas':alphas,'load': {'mean':True,'std':True,'s':True} })
    exps.append({'ii':62,'order':30,'alphas':alphas,'load': {'mean':True,'std':True,'s':True} })
    exps.append({'ii':63,'order':30,'alphas':alphas,'load': {'mean':True,'std':True,'s':True} })

    # 17..22
    def getAlpha(x):
        alphas = np.ones(16)
        alphas[6:] = x
        return alphas

    for ee in [0.0,0.2,0.4,0.6,0.8,1.0]:
        exps.append({'ii':10,'order':3,'alphas':getAlpha(ee),'load': {'mean':True,'std':True,'s':True} })

    # 23..28
    def getAlpha(x):
        alphas = np.ones(16)
        alphas[:] = x
        return alphas

    for ee in [0.0,0.2,0.4,0.6,0.8,1.0]:
        exps.append({'ii':10,'order':20,'alphas':getAlpha(ee),'load': {'mean':True,'std':True,'s':True} })


    #exps.append({'ii':8,'order':20,'alphas':np.ones(16)*0.8,'load': {'mean':False,'std':False,'s':False} })

    do_exps = [13,14,15,16] # in paper
    do_exps = [17,18,19,20,21,22] # new exp. for paper
    do_exps = range(23,28+1)

    def remove_exp_files(num):
        for f in glob('res/exp_%d*'%num):
            remove(f)

    # create data is it doesn't exist
    m_file_name = 'm_train_other.bin'
    if not exists(m_file_name):
        syndata = SynData(dataset='other')
        print 'GENERATING DATA'
        syndata.save_step()

    for exp_no,exp in enumerate(exps):
        #exp = exps[exp_no]

        if exp_no not in do_exps:
            continue
        print '######## CUR EXP',exp_no, '#########'
        # remove previously saved files.
        remove_exp_files(exp_no)

        #syndata = SynData(dataset='kth')
        syndata = SynData(dataset='other')

        # step 1 - load data
        syndata.load_step()

        # step 2 - analyse and save new
        #MuAnalysis()

        # steps 3+ - save new G and synthesize
        stats_src, stats_tar = syndata.save_new_G(exp['ii'],exp['order'],exp['alphas'],exp['load'],exp_no=exp_no)

        # synthesize
        texture = Texture(exp_no=exp_no,stats = [stats_src, stats_tar])

        # syn fbm
        #use_fbm=True
        #im = getIm(use_fbm)
        #texture.synTexture(im=im,G0_from='new_G.bin',onlyGram = False)

        # syn from modified G
        plt.close('all')
        stats_src, stats_tar, stats_im, res_im = texture.synTexture(im=None,G0_from='new_G.bin',onlyGram = False,maxiter=230)

        print 'saving experiment stats'
        exp_stats = {}
        exp_stats['src'] = stats_src
        exp_stats['tar'] = stats_tar
        exp_stats['syn'] = stats_im
        exp_stats['res_im'] = res_im
        exp_stats['alphas'] = exp['alphas']
        pickle.dump(exp_stats,open('res/exp_%d_stats.bin'%exp_no,'w'))

    """
    original_dim = 224**2
    n=200
    x_train0, x_test0, y_train0, y_test0 = \
        get_kth_imgs(N=50000,n=n,reCalc=False,resize=original_dim)
    im = x_train0[0]
    im = np.stack([im,im,im],axis=2).astype(np.float32) # grayscale

    #use_fbm=False
    #im = getIm(use_fbm)

    do_synth=True
    im0=im

    #texture.synTexture(im)

    ##
    if do_synth:
        texture.synTexture(im,G0_from='new_G.bin',onlyGram = False)
    else: # save G0
        ##
        g0_fname = 'G0.bin'
        reCalc=True
        if not exists(g0_fname) or reCalc:
            G0 = synTexture(im,onlyGram = True)
            pickle.dump(G0,open(g0_fname,'w'))
            print 'saved G0 to file', g0_fname
        else:
            print 'loading G0 from file', g0_fname
            G0 = pickle.load(open(g0_fname,'r'))
        print G0
    """
