import os
from os import listdir
import matplotlib.pyplot as plt
import cPickle as pickle
import numpy as np
from fbm2d import synth2, hurst2d
from fbm.fbm import fbm
from sklearn.model_selection import train_test_split
from PIL import Image
from scipy.misc import imresize

import tensorflow as tf
import keras.backend as K

def generate_1d_fbms(N=10000,n=256,reCalc=False):
    #reCalc = True
    #reCalc = False
    if os.path.exists('data1d.bin') and not reCalc:
        print 'loading data from file'
        Xtrain, Xtest, Ytrain, Ytest = pickle.load(open('data1d.bin','r'))
    else:
        print 'generating data and saving'
        np.random.seed(0)
        X = []
        H = np.linspace(0.05,0.8,N)# [0.5]*N
        for i in H:
            fbmr,fgnr,times= fbm(n-1,i,L=1)
            fbmr = fbmr-np.min(fbmr)
            fbmr = fbmr/np.max(fbmr)
            X.append(fbmr)
        X0=np.array(X)
        H = np.array(H)
        Y = H
        X=np.expand_dims(X0,2) # to make input of size (n,1) instead of just (n)
        Xtrain, Xtest, Ytrain, Ytest = train_test_split(X0,Y,test_size=0.2,random_state=0)

        pickle.dump([Xtrain,Xtest,Ytrain,Ytest],open('data1d.bin','w'))

def generate_2d_fbms(N=10000,n=32,reCalc=False,resize=None,noise=False):
    # noise indicates whether or not to include the originating noise with the samples (for cvaes)
    #reCalc = True
    #reCalc = False
    if noise:
        fname = 'data2dn.bin'
    else:
        fname = 'data2d.bin'

    if os.path.exists(fname) and not reCalc:
        print 'loading data from file'
        Xtrain, Xtest, Ytrain, Ytest = pickle.load(open(fname,'r'))
    else:
        print 'generating data and saving'
        np.random.seed(0)
        X = []
        H = np.linspace(0.05,0.8,N)# [0.5]*N
        for k,i in enumerate(H):
            if not k % 100:
                print 'done %f'%(1.0*k/N)
            #fbmr,fgnr,times= fbm(n-1,i,L=1)
            fbm,noises = synth2(N=n,H=i)
            fbm = fbm-np.min(fbm)
            fbm = fbm/np.max(fbm)

            if noise:
                #noises = [np.reshape(x,[1,-1]) for x in noises]
                #noises = np.hstack(noises)
                X.append(fbm)
            else:
                X.append(fbm)
        X0=np.array(X)
        H = np.array(H)
        Y = H
        #X=np.expand_dims(X0,2) # to make input of size (n,1) instead of just (n)
        if resize is not None:
            sz=int(np.sqrt(resize))
            X0r = np.zeros([X0.shape[0],sz,sz])
            for i in range(X0.shape[0]):
                X0r[i,:,:] = imresize(X0[i,:,:],[sz,sz])
                #print X0r[i,:,:]
                #plt.imshow(X0r[i,:,:],interpolation='none',cmap='gray')
                #plt.show()
            X0=X0r


        Xtrain, Xtest, Ytrain, Ytest = train_test_split(X0,Y,test_size=0.2,random_state=0)

        pickle.dump([Xtrain,Xtest,Ytrain,Ytest],open(fname,'w'))

    return Xtrain, Xtest, Ytrain, Ytest

class GetOutOfLoop( Exception ):
    pass

def get_kth_imgs(N=10000,n=32,reCalc=False,resize=None):
    #reCalc = True
    #reCalc = False

    if os.path.exists('data_kth.bin') and not reCalc:
        print 'loading data from file'
        Xtrain, Xtest, Ytrain, Ytest = pickle.load(open('data_kth.bin','r'))
    else:
        print 'loading data and saving'
        X = []
        H = []
        names = []
        path0 = '/home/ido/combined_model_classification/KTH-TIPS2-b/wool/'
        path = ['sample_', '/']

        dirs = ['a','b','c','d']
        files = []
        for d in dirs:
            path1 = path0+path[0]+d+path[1]
            files += [path1+f for f in listdir(path1)]
        print 'tot files:',len(files)
        p = n
        try:
            for i,f in enumerate(files):
                print i,': file',f
                img = Image.open(f).convert('L')
                (width, height) = img.size
                img = np.array(list(img.getdata()))
                img = img.reshape((height, width))
                sz = [width, height]

                for k in range(0,sz[0],p):
                    for l in range(0,sz[1],p):
                        patch = img[k:k+p,l:l+p]*1.0
                        patch = patch-np.min(patch)
                        patch = patch/np.max(patch)
                        #print patch
                        if patch.shape[0]<p or patch.shape[1]<p:
                            continue
                        h,_ = hurst2d(patch,max_tau=7)
                        H.append(h)
                        names.append(f)
                        X.append(patch)
                        if len(X)>=N:
                            print 'too many images'
                            raise GetOutOfLoop
        except GetOutOfLoop:
            pass
        print 'max N',N,'number of patches',len(X)
        Y = H#np.array(H)
        X = np.array(X)
        if resize is not None:
            sz=int(np.sqrt(resize))
            X0r = np.zeros([X.shape[0],sz,sz])
            for i in range(X.shape[0]):
                X0r[i,:,:] = imresize(X[i,:,:],[sz,sz])
                #print X0r[i,:,:]
                #plt.imshow(X0r[i,:,:],interpolation='none',cmap='gray')
                #plt.show()
            X=X0r
        #X = np.expand_dims(np.array(X),3) # to make input of size (n,1) instead of just (n)
        Xtrain, Xtest, Ytrain, Ytest = train_test_split(X,Y,test_size=0.2,random_state=0)

        pickle.dump([Xtrain,Xtest,Ytrain,Ytest],open('data_kth.bin','w'))

    return Xtrain, Xtest, Ytrain, Ytest


def loss_logvarinc(y_true, y_pred, n, taus=[0,1,2,3]):
    """Need tf0.11rc to work"""
    #y_true = tf.reshape(y_true, [batch_size] + get_shape(y_pred)[1:])
    #y_pred = tf.reshape(y_pred, [batch_size] + get_shape(y_pred)[1:])
    #y_true = K.expand_dims(y_true,3)
    #y_pred = K.expand_dims(y_pred,3)
    #dif = [tf.constant([ [1,-1] ])]
    #dif.append(tf.constant([ [1,0,-1] ]))
    #dif.append(tf.constant([ [1,0,0,-1] ]))
    #dif.append(tf.constant([ [1,0,0,0,-1] ]))
    #dif.append(tf.constant([ [1,0,0,0,0,-1] ]))
    def lvi(y,f):

        m = f
        f = tf.concat([tf.constant([[1.0]]), tf.zeros([1,m]), tf.constant([[-1.0]])],axis=1)
        #f = tf.Print(f,[f[0]],'ff')
        #print 'f',f

        #if trans:
        #    f = tf.transpose(f)
        def get1(f1):

            f = tf.expand_dims(f1,2)
            f = tf.expand_dims(f,3)

            #print 'fa',f,'y',y
            y1 = tf.expand_dims(y,3)
            f1 = tf.nn.convolution(y1,f,'VALID')

            m1 = m + 2
            f1 = tf.reshape(f1,[-1,(n-m1+1)*n])

            f1 = K.var(f1,axis=1)
            f1 = tf.log(1+f1)
            return f1

        f1 = get1(f) + get1(tf.transpose(f))
        print 'f1',f1
        res = f1/2.0

        res.set_shape(y[:,0,0].get_shape())
        return res#f1/2.0#[0]

    lvi1_true = lambda _, f : lvi(y_true,f)
    lvi1_pred = lambda _, f : lvi(y_pred,f)
    dif = tf.constant(taus)
    #print 'y',y[:,0,0]
    lvi_true = tf.scan(lvi1_true,dif,initializer=tf.zeros_like(y_true[:,0,0]),infer_shape=False)
    lvi_pred = tf.scan(lvi1_pred,dif,initializer=tf.zeros_like(y_true[:,0,0]),infer_shape=False)
    #lvi_pred = tf.Print(lvi_pred,[lvi_pred[2]])
    lvi_true = tf.transpose(lvi_true)
    lvi_pred = tf.transpose(lvi_pred)
    print 'lvi_pred',lvi_pred
    res = K.mean(K.square(lvi_true-lvi_pred),axis=1)
    #res = tf.Print(res,[res],'lvi')
    return res


def loss_DSSIS_tf11(y_true, y_pred, blur=True):
    """Need tf0.11rc to work"""
    #y_true = tf.reshape(y_true, [batch_size] + get_shape(y_pred)[1:])
    #y_pred = tf.reshape(y_pred, [batch_size] + get_shape(y_pred)[1:])
    y_true = K.expand_dims(y_true,3)
    y_pred = K.expand_dims(y_pred,3)

    #print blurfilt
    if blur:
        filtsz=5
        blurfilt = tf.ones([filtsz,filtsz])/(1.0*filtsz**2)
        blurfilt = tf.expand_dims(blurfilt,2)
        blurfilt = tf.expand_dims(blurfilt,3)

        y_true = tf.nn.convolution(y_true,blurfilt,'VALID')
        y_pred = tf.nn.convolution(y_pred,blurfilt,'VALID')
    print y_pred
    #y_true = tf.transpose(y_true, [0, 2, 3, 1])
    #y_pred = tf.transpose(y_pred, [0, 2, 3, 1])
    pad_method = 'VALID'
    w = 10
    patches_true = tf.extract_image_patches(y_true, [1, w, w, 1], [1, 1, 1, 1], [1, 1, 1, 1], pad_method)
    patches_pred = tf.extract_image_patches(y_pred, [1, w, w, 1], [1, 1, 1, 1], [1, 1, 1, 1], pad_method)
    print 'patches_pred',patches_pred
    u_true = K.mean(patches_true, axis=3)
    u_pred = K.mean(patches_pred, axis=3)
    var_true = K.var(patches_true, axis=3)
    var_pred = K.var(patches_pred, axis=3)
    std_true = K.sqrt(var_true)
    std_pred = K.sqrt(var_pred)

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    ssim = (2 * u_true * u_pred + c1) * (2 * std_pred * std_true + c2)
    denom = (u_true ** 2 + u_pred ** 2 + c1) * (var_pred + var_true + c2)
    ssim /= denom
    ssim = tf.where(tf.is_nan(ssim), K.zeros_like(ssim), ssim)

    ssim = tf.reduce_mean(ssim,axis=1)
    ssim = tf.reduce_mean(ssim,axis=1)
    ssim1 = (1.0-ssim)/2.0 # K.mean(((1.0 - ssim) / 2),axis=1)
    #ssim1 = tf.Print(ssim1,[ssim1],'SSIM')
    return ssim1





if __name__=='__main__':
    #    generate_2d_fbms(N=200,n=128,reCalc=True,resize=64**2)
    #x = tf.get_variable('x',shape=(5,16,16))
    #y = tf.get_variable('y',shape=(5,16,16))
    #res = loss_DSSIS_tf11(x,y)
    x = tf.Variable(tf.random_normal((7,16,16)))
    y = tf.Variable(tf.random_normal((7,16,16)))
    res = loss_logvarinc(x,y,16,taus=[0,1])
    print 'res', res
    sess=tf.Session()
    sess.run(tf.global_variables_initializer())
    print sess.run(res)


