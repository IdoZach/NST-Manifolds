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
from numpy.linalg import svd, matrix_rank
from scipy.linalg import svd as ssvd
from scipy.sparse.linalg import svds
from sklearn.decomposition import PCA
from vgg_19_keras import Texture
from fbm_data import synth2, get_kth_imgs
from keras.models import load_model, save_model

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
        print encoded
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

    def get_G_kth(self,texture, im):
        #texture = Texture()

        #saved_G = 'KTH_G.bin'
        #saved_F = 'KTH_F.bin'
        #gg = []
        #ff = []

        im = np.stack([im,im,im],axis=2).astype(np.float32) # grayscale
        _, F0 = texture.synTexture(im,onlyGram = True)
        return F0

    def save_svds(self,F,lengths=[2,2,4,4,4],k=20):
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
                print 'calc svd inner %d'%l,
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
        #pickle.dump([UU,ss,VV,SS],open('svd_res_%d.bin'%ii,'w'))
        print 'done'
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
            desc_out = desc
        if do_load:
            desc_out = pickle.load(open('desc_%d.bin'%load_i,'r'))
        pickle.dump(new_G,open('new_G.bin','w'))

        return new_G, desc_out

    def prepare_data(self,idxs=None):
        x_train0, x_test0, y_train0, y_test0 = \
            get_kth_imgs(N=50000,n=self.n,reCalc=False,resize=self.original_dim)
        x_train0[0],_ = synth2(224,H=0.3)
        x_train0[1],_ = synth2(224,H=0.5)
        for i in range(2):
            x_train0[i]-=np.min(x_train0[i]*1.0)
            x_train0[i]=x_train0[i]/np.max(x_train0[i])*256.0
        print 'two first samples are fBm'
        if idxs is not None:
            return x_train0[idxs], x_test0, y_train0[idxs], y_test0
        else:
            return x_train0, x_test0, y_train0, y_test0

    def __init__(self, step_save=False, use_ae=False):
        ### here we run with 'save_step=True' to save data.
        ### then we run with 'False' to save the encoded information and G matrices
        ### this can be used in mu_analysis.py to analyse and in vgg_19_keras.py to generate
        print 'creating syn object'
        self.step_save = False#True
        self.use_ae = False
        self.original_dim = 224**2
        self.n=200
        self.weights = np.array([16,16,4,4,2,2,2,2,1,1,1,1,1,1,1,1],dtype=np.float32) # scaling of means according to inverse matrix size

    def save_step(self):
        # prepare training data
        x_train0, x_test0, y_train0, y_test0 = prepare_data()
        # process
        m_train = []
        m_test = []
        texture = Texture()
        def get_desc(im):
            F0 = get_G_kth(texture,im)
            UsVS = save_svds(F0)
            _, desc = processF(UsVS,save_i=0,load_std=False)
            return desc

        print 'saving results...'

        for i, im in enumerate(x_train0):
            print 'train', i+1, '/', len(x_train0)
            desc=get_desc(im)
            m_train.append(desc)
            pickle.dump(m_train,open('m_train.bin','w'))

        for i, im in enumerate(x_test0):
            print 'test', i+1, '/', len(x_test0)
            desc=get_desc(im)
            m_test.append(desc)
            pickle.dump(m_test,open('m_test.bin','w'))

        pickle.dump([y_train0,y_test0],open('m_y.bin','w'))
        return m_train, m_test, y_train0, y_test0

    def load_step(self):
        print 'loading data and preprocessing train_vars.bin'
        m_train = pickle.load(open('m_train.bin','r'))
        m_test = pickle.load(open('m_test.bin','r'))
        y_train0, y_test0 = pickle.load(open('m_y.bin','r'))

        choose_classes = ['woola','woolb','woolc','woold']
        choose_classes = []
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

        train_vars = []
        test_vars = []
        train_vars_std = []
        test_vars_std = []
        for i in range(len(train_means[0])): # 16
            train_vars.append( np.array([x[i] for x in train_means ]))
            test_vars.append( np.array([x[i] for x in test_means ]))
            train_vars_std.append( np.array([x[i] for x in train_stds ]))
            test_vars_std.append( np.array([x[i] for x in test_stds ]))
        train_vars = [t*w for t,w in zip(train_vars,weights)]
        test_vars = [t*w for t,w in zip(test_vars,weights)]

        # try pca for the entire dataset
        pickle.dump([train_vars, train_vars_std],open('train_vars.bin','w'))
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
        return train_vars, train_vars_std

    def save_new_G(self):
        print 'saving new G for synthesis'
        x_train0, x_test0, y_train0, y_test0 = self.prepare_data(self.sel_indexes)

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

        ii=2
        alpha=0.7

        if use_ae:
            print 'predicting all training set'
            all_pred = AE.ae.predict(train_vars)
            all_enc = AE.encoder.predict(train_vars)
            print 'saving in and out of sample',ii
            train_one = [ np.array([t[ii]]) for t in train_vars ]
            desc_mean_out = AE.ae.predict(train_one)
            desc_mean_out = [1.0*t/w for t,w in zip(desc_mean_out,weights)]

            # apply distortion on latent dimension
            one_encoded = AE.encoder.predict(train_one)
            #i_source = 15
            print 'enc',one_encoded
            one_encoded, i_source, explained_variance = interpolate_latent(all_enc,one_encoded,
                                                                       i_source=ii,
                                                                       alpha=alpha)
            desc_mean_out_modified = AE.decoder.predict(one_encoded)

        # using pca instead of AE+pca
        load_file = 'train_vars.bin'
        #load_file = 'train_vars_comp.bin'
        print 'loading from file',load_file
        train_vars_compressed, train_vars_std_compressed = pickle.load(open(load_file,'r'))

        i2 = 10
        i_source = i2
        alphas = np.ones(16)*0.0
        #alphas[6:]=1.0
        alphas[8:] = 0.0 # move everything to the other texture
        desc_mean_out_modified = [ np.array([t[ii]]) for t in train_vars_compressed ]
        desc_mean_out_modified2 = [ np.array([t[i2]]) for t in train_vars_compressed ]
        desc_mean_out_modified = [ a*g+(1-a)*h for a,g,h in
                                      zip(alphas,desc_mean_out_modified,desc_mean_out_modified2)]

        desc_std_out = [ np.array([t[ii]]) for t in train_vars_std_compressed ]
        desc_mean_out = [1.0*t/w for t,w in zip(desc_mean_out_modified,weights)]
        print 'using modified latent variables'

        # postprocess
        def postprocess(m):
            out=[]
            for mean in m:
                mean[mean<0]=0.0
                out.append(mean)
            return out
        desc_mean_out = postprocess(desc_mean_out)
        new_desc = m_train[ii].copy() # point 1 for interpolation
        source_desc = m_train[i_source].copy() # point 2 for interpolation
        desc_comparison = {'old':new_desc.copy()}
        new_desc['mean'] = desc_mean_out
        # interpolate also stds
        mul = lambda x,y,a: [a*g+(1-a)*h for g,h in zip(x,y)]
        new_desc['mean'] = source_desc['mean']
        new_desc['std'] = source_desc['std']
        desc_comparison['new'] = new_desc

        plt.hold(False)
        i=1
        if use_ae:
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

        texture=Texture()
        if use_ae:
            pickle.dump([all_pred, all_enc, y_train0],open('all_pred.bin','w'))
        print 'saving images...'
        plt.imsave('cur_im_src.png',x_train0[i_source]/255.0,cmap=plt.cm.gray)
        plt.imsave('cur_im.png',x_train0[ii]/255.0,cmap=plt.cm.gray)

        F0 = self.get_G_kth(texture,x_train0[ii])
        UsVS = self.save_svds(F0)
        self.new_G, self.desc_out = self.processF(UsVS,load_i=ii,load_std=False) # this uses desc file and saves new_G to be used by vgg_19_keras.py file

    def getParams(self):
        return self.new_G, self.desc_out

if __name__=='__main__':
    syndata = SynData()
    syndata.load_step()

    syndata.save_new_G()

