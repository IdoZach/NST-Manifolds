'''This script demonstrates how to build a variational autoencoder with Keras.

Reference: "Auto-Encoding Variational Bayes" https://arxiv.org/abs/1312.6114
'''
import tsne
import os
import cPickle as pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import mlab
from fbm.fbm import fbm
from fbm2d import synth2, hurst2d
from sklearn.model_selection import train_test_split

from scipy.stats import norm
from scipy.stats import linregress
from keras.layers import Input, Dense, Lambda, MaxPool2D, \
    Conv2D, Flatten, Conv2DTranspose, RepeatVector, LSTM, Reshape # there's also convlstm2d
from keras.models import Model
from keras import backend as K
from keras import metrics
from keras.utils import plot_model
from keras.datasets import mnist


def hurst(ts):

    """Returns the Hurst Exponent of the time series vector ts"""
    # Create the range of lag values
    lags = range(2, 30)

    # Calculate the array of the variances of the lagged differences
    # Here it calculates the variances, but why it uses
    # standard deviation and then make a root of it?
    tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]

    # Use a linear fit to estimate the Hurst Exponent
    p0, p1, rval, _, _ = linregress(np.log(lags),np.log(tau))
    #poly, _, _, _, rcond = np.polyfit(np.log(lags), np.log(tau), 1, full=True)

    # Return the Hurst exponent from the polyfit output
    return p0*2.0, rval**2

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

def generate_2d_fbms(N=10000,n=32,reCalc=False):
    #reCalc = True
    #reCalc = False
    if os.path.exists('data2d.bin') and not reCalc:
        print 'loading data from file'
        Xtrain, Xtest, Ytrain, Ytest = pickle.load(open('data2d.bin','r'))
    else:
        print 'generating data and saving'
        np.random.seed(0)
        X = []
        H = np.linspace(0.05,0.8,N)# [0.5]*N
        for k,i in enumerate(H):
            if not k % 100:
                print 'done %f'%(1.0*k/N)
            #fbmr,fgnr,times= fbm(n-1,i,L=1)
            fbm = synth2(N=n,H=i)
            fbm = fbm-np.min(fbm)
            fbm = fbm/np.max(fbm)
            X.append(fbm)
        X0=np.array(X)
        H = np.array(H)
        Y = H
        X=np.expand_dims(X0,2) # to make input of size (n,1) instead of just (n)
        Xtrain, Xtest, Ytrain, Ytest = train_test_split(X0,Y,test_size=0.2,random_state=0)

        pickle.dump([Xtrain,Xtest,Ytrain,Ytest],open('data2d.bin','w'))

    return Xtrain, Xtest, Ytrain, Ytest

## load data
n=16
Xtrain, Xtest, Ytrain, Ytest = generate_2d_fbms(N=50000,n=n,reCalc=False)

## ################################################
# train and learn network

batch_size = 200
original_dim = n
latent_dim = 3
intermediate_dim = n**2/4
intermediate_dim2 = n**2/8
intermediate_dim3 = n**2/16
epochs = 20
epsilon_std = 1.0
n_units = n**2
img_rows, img_cols, img_chns = n, n, 1
filters = 64
num_conv = 3
original_img_size = (img_rows, img_cols, img_chns)
output_shape = (batch_size, 14, 14, filters)

x = Input(shape=(original_dim,original_dim))
levels = 5
def scale_space(x):

    # Shape = 1 x height x width x 1.
    x_expanded = K.expand_dims(x, 2)
    x_expanded = K.expand_dims(x_expanded, 1)
    #x_expanded = K.repeat_elements(x_expanded, 5, axis=3)
    #x_expanded = K.permute_dimensions(x_expanded,[0,2,1])
    print x_expanded
    levels = 5#x_expanded.shape[1]
    filters = [gauss2D_filter(sigma=0.1)] # delta filter
    sigmas = np.linspace(0.5,2.5,levels)
    for s in sigmas:
        filt = gauss2D_filter(sigma=s)
        #filters.append(np.convolve(filters[-1],filter,mode='same'))
        filters.append(filt-filters[-1])
    filters=filters[:-1]
    filters = np.array(filters)
    #print filters.shape
    Kfilters = K.constant(np.transpose(filters))
    Kfilters = K.expand_dims(Kfilters,2)
    Kfilters = K.expand_dims(Kfilters,3)
    Kfilters = K.permute_dimensions(Kfilters,[0,2,3,1])
    #print Kfilters
    #print x_expanded
    x = K.conv2d(x_expanded,Kfilters,padding='same')
    x = K.squeeze(x,axis=1)
    x = K.permute_dimensions(x,[0,2,1])
    return x

max_ss_sigma = 2.5

def scale_space_2d(x,invert=False):

    # Shape = 1 x height x width x 1.
    x_expanded = K.expand_dims(x, 3)
    #x_expanded = K.expand_dims(x, 1)

    print x_expanded
    levels = 5
    filters = [gauss2D_filter(sigma=0.1,one_D=False)] # delta filter
    sigmas = np.linspace(0.5,max_ss_sigma,levels)
    for s in sigmas:
        filt = gauss2D_filter(sigma=s,one_D=False)
        # difference of Gaussians
        filters.append(filt-filters[-1])
    #filters=filters[:-1]# exclude last filter
    filters=filters[1:]# exclude first filter
    if invert:
        filters = filters[::-1]
    filters = np.array(filters)
    #print filters.shape
    #Kfilters = K.constant(np.transpose(filters))
    #Kfilters = K.expand_dims(Kfilters,2)
    Kfilters = K.expand_dims(filters,3)
    # filter is [height width in_channels out_channels]
    Kfilters = K.permute_dimensions(Kfilters,[1,2,3,0])
    print Kfilters
    #print x_expanded
    x = K.conv2d(x_expanded,Kfilters,padding='same')
    #x = K.squeeze(x,axis=1)

    # now we're flattening the vector, probably not the best thing to do.
    x = K.reshape(x,[-1,n**2,levels])
    x = K.permute_dimensions(x,[0,2,1])
    print 'x',x
    #x = K.permute_dimensions(x,[0,2,1])

    return x # last is the coarsest

# deterministic part modelling
x1 = Lambda( lambda x: K.expand_dims(x,3) )(x)

conv_1 = Conv2D(img_chns,
                kernel_size=(2, 2),
                padding='same', activation='relu')(x1)
conv_2 = Conv2D(filters,
                kernel_size=(2, 2),
                padding='same', activation='relu',
                strides=(2, 2))(conv_1)
conv_3 = Conv2D(filters,
                kernel_size=num_conv,
                padding='same', activation='relu',
                strides=1)(conv_2)
flat = Flatten()(conv_3)
hidden = Dense(intermediate_dim, activation='relu')(flat)

z_mean = Dense(latent_dim)(hidden)
z_log_var = Dense(latent_dim)(hidden)


def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(batch_size, latent_dim),
                              mean=0., stddev=epsilon_std)
    return z_mean + K.exp(z_log_var) * epsilon

z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])
#
n1=8
output_shape = (batch_size, n1, n1, filters)
decoder_hid = Dense(intermediate_dim, activation='relu')
decoder_upsample = Dense(filters * n1 * n1, activation='relu')


decoder_reshape = Reshape(output_shape[1:])
decoder_deconv_1 = Conv2DTranspose(filters,
                                   kernel_size=num_conv,
                                   padding='same',
                                   strides=1,
                                   activation='relu')
decoder_deconv_2 = Conv2DTranspose(filters, num_conv,
                                   padding='same',
                                   strides=1,
                                   activation='relu')
output_shape = (batch_size, n+1, n+1, filters)

decoder_deconv_3_upsamp = Conv2DTranspose(filters,
                                          kernel_size=(3, 3),
                                          strides=(2, 2),
                                          padding='valid',
                                          activation='relu')
decoder_mean_squash = Conv2D(img_chns,
                             kernel_size=2,
                             padding='valid',
                             activation='sigmoid')

def decode_pipeline(z):
    hid_decoded = decoder_hid(z)
    up_decoded = decoder_upsample(hid_decoded)
    reshape_decoded = decoder_reshape(up_decoded)
    deconv_1_decoded = decoder_deconv_1(reshape_decoded)
    deconv_2_decoded = decoder_deconv_2(deconv_1_decoded)
    x_decoded_relu = decoder_deconv_3_upsamp(deconv_2_decoded)
    x_decoded_mean = decoder_mean_squash(x_decoded_relu)
    x_decoded_mean = Lambda( lambda x: K.squeeze(x,3))(x_decoded_mean)
    return x_decoded_mean



# random part modelling
print 'x...', x
y_hp = Lambda(scale_space_2d)(x)
print 'y_hp',y_hp

shared_LSTM = LSTM(units=n_units,return_sequences=False)
yr = shared_LSTM(y_hp) # this should be the generating noise
print 'yr',yr # now this should actually be some noise factor

noise_dim = 256
z2_mean = Dense(levels * noise_dim,activation='relu')(yr)
z2_log_var = Dense(levels * noise_dim,activation='relu')(yr)

levels = 5
def sampling_hp(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(batch_size, levels*noise_dim),
                              mean=0., stddev=epsilon_std)
    res = K.reshape(z_mean + K.exp(z_log_var)*epsilon,(batch_size,levels,noise_dim))
    return res#z_mean + K.exp(z_log_var) * epsilon

print 'built layers'
z_noise = Lambda(sampling_hp, output_shape=(levels,noise_dim,))([z2_mean, z2_log_var])
print 'z noise', z_noise
# decoding hp
decoder_hp_rnn = shared_LSTM # LSTM(units=n_units)
#print 'decoder_rnn',decoder_rnn
#summation = Lambda(lambda x: K.sum(x,axis=1)) # not sure about the axis
decoder_hp_fc = Dense(original_dim**2)

def decode_pipeline_full(z, z_noise):
    x_decoded_lp_mean = decode_pipeline(z)
    print 'before'
    print z_noise
    dec = decoder_hp_rnn(z_noise)
    print 'after'
    dec = decoder_hp_fc(dec)
    dec = Reshape((n,n))(dec)

    dec = Lambda(lambda x: x + x_decoded_lp_mean)(dec)
    return dec

x_decoded_mean = decode_pipeline_full(z,z_noise)
#print 'decoded', x_decoded_mean
def vae_loss(x, x_decoded_mean):
    #x_decoded_mean, x_decoded_log_std = x_dec
    #xent_loss = original_dim * metrics.binary_crossentropy(x, x_decoded_mean)
    #print x_decoded_mean
    #print x_decoded_log_std
    #term1 = K.square(x-x_decoded_mean)/(2.0 * K.exp(x_decoded_log_std))
    #term2 = 0.5 * K.sum(x_decoded_log_std)
    #reconstruction_loss = original_dim*(term1 + term2)
    term1 = K.flatten(x-x_decoded_mean)
    term1 = K.mean(K.square(term1))
    #reconstruction_loss  = original_dim**2 * metrics.mean_squared_error(x, x_decoded_mean)
    reconstruction_loss  = original_dim**2 * term1
    print 'rec loss',reconstruction_loss
    kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    kl_loss2 = - 0.5 * K.sum(1 + z2_log_var - K.square(z2_mean) - K.exp(z2_log_var), axis=-1)
    #kl_loss = - 0.5 * K.sum(1 + K.log(0.1+z_log_var) - K.square(z_mean) - z_log_var, axis=-1)
    #return xent_loss + kl_loss
    return reconstruction_loss + kl_loss + kl_loss2

print 'before compiling, x',x,'x dec',x_decoded_mean
vae = Model(x, outputs=x_decoded_mean)
vae.compile(optimizer='RMSprop', loss=vae_loss) # was RMSprop

# train the VAE on MNIST digits
print 'after compiling'
#(x_train, y_train), (x_test, y_test) = mnist.load_data()


#x_train = x_train.astype('float32') / 255.
#x_test = x_test.astype('float32') / 255.
#x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
#x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

vae.fit(Xtrain, Xtrain,
        shuffle=True,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(Xtest, Xtest))
print 'done training.'
# build a model to project inputs on the latent space
encoder = Model(x, z_mean)

# display a 2D plot of the digit classes in the latent space
x_test_encoded = encoder.predict(Xtest, batch_size=batch_size)

########################################################################################################################
# see the histograms of latent variables
plt.hold(True)
f, spl = plt.subplots(latent_dim)
for s,var in zip(spl,x_test_encoded.T):


    _, bins, patches = s.hist(var, 50, normed=1, facecolor='green', alpha=0.75)
    # add a 'best fit' line
    y = mlab.normpdf( bins, 0, 1)
    l = s.plot(bins, y, 'r--', linewidth=1)
########################################################################################################################
#plt.figure(figsize=(6, 6))
# see correspondence between true latents and estimated
if x_test_encoded.shape[1]>=2:
    plt.clf(); plt.hold(False)
    def cart2pol(x, y):
        rho = np.sqrt(x**2 + y**2)
        phi = np.arctan2(y, x)
        return(rho, phi)
    polvec = cart2pol(x_test_encoded[:,0],x_test_encoded[:,2])
    plt.scatter(polvec[0],polvec[1], c=Ytest, linewidths=0.1,s=10)
    plt.xlabel('radius'),plt.ylabel('angle')
    #plt.colorbar()
    rad = np.sqrt(np.sum(np.power(x_test_encoded[:,0:2],2),axis=1))
    #pass
    poly = np.polyfit(rad,Ytest,deg=1)
    #plt.scatter(rad,Ytest)
    #plt.xlabel('z1'),plt.ylabel('z2')
else:
    plt.scatter(x_test_encoded[:,0],Ytest)#, c=Ytest)

plt.show()
#########################################################################################################################
## show manifold with t-SNE - takes a long time and doesn't seem to work too well

SNE = tsne.tsne(x_test_encoded, 2, 10, 20.0);
plt.scatter(SNE[:,0], SNE[:,1], 20, Ytest);

########################################################################################################################
## sample
decoder_input = Input(shape=(latent_dim,))
decoder_noise_input = Input(shape=(levels,noise_dim,))
#_z_expanded = Lambda( lambda x: K.expand_dims(K.expand_dims(x,1),2) )(decoder_input)
_x_decoded_mean = decode_pipeline_full(decoder_input,decoder_noise_input)
#_h_decoded = decoder_h2(decoder_h1(decoder_input))
#_x_decoded_mean = decoder_mean(_h_decoded)
generator = Model([decoder_input, decoder_noise_input], _x_decoded_mean)

# display a 2D manifold of the digits
m = 16  # figure with 15x15 digits
# linearly spaced coordinates on the unit square were transformed through the inverse CDF (ppf) of the Gaussian
# to produce values of the latent variables z, since the prior of the latent space is Gaussian
plt.ion()
plt.hold(False)
save_H = [False]*10
H_vals = np.linspace(0.2,0.8,len(save_H))
H_saved = {}
plt.close()
_, spl = plt.subplots(3)

plt.hold(True)
tot=0; rsquares=0
Hs = []
rrs = []
num_exps=1000
for i in range(num_exps):
    #z_sample = np.array([[xi, yi, zi]])
    rr = np.random.randn((latent_dim))*2
    rrs.append(rr)
    #rr[0]=0
    #z_sample = np.array([[xi, yi, zi, rr[0], rr[1]]])
    z_sample = np.array([rr])
    z_noise_sample = np.random.randn(levels,noise_dim)
    z_noise_sample = np.array([z_noise_sample])
    x_decoded = generator.predict([z_sample,z_noise_sample])
    x_decoded = np.reshape(x_decoded[0],[n,n])
    """
    plt.imshow(x_decoded,interpolation='none')
    plt.show()
    plt.pause(0.1)
    continue
    #"""

    #digit = x_decoded[0].reshape(digit_size, digit_size)
    x_decoded = x_decoded-np.min(x_decoded)
    x_decoded = x_decoded/np.max(x_decoded)
    H_est, rsquare = hurst2d(x_decoded,max_tau=7)
    Hs.append(H_est)
    rsquares+=rsquare
    tot+=1
    if not i % 30:
        spl[0].hold(False)
        diffres = np.squeeze(np.reshape(x_decoded[1:]-x_decoded[:-1],[1,-1]))
        _, bins, patches = spl[0].hist(diffres, 50, normed=1, facecolor='green', alpha=0.75)
        y = mlab.normpdf( bins, np.mean(diffres), np.std(diffres))
        spl[0].hold(True)
        spl[0].plot(bins, y, 'r--', linewidth=1)
        spl[0].set_title('diff, exp %d/%d'%(i,num_exps))
        spl[1].hold(False)
        spl[1].imshow(x_decoded,interpolation='none',cmap='gray')

        #plt.title('y %f, x %f, z %f'%(yi,xi,zi))

        if len(Hs)>10:
            spl[2].hold(False)
            spl[2].hist(np.array(Hs),10)

        plt.title('H %f r2 %f mean r2 %f'%(H_est,rsquare,rsquares/tot))
        plt.show()
        plt.pause(0.1)

#plt.figure(figsize=(10, 10))
#plt.imshow(figure, cmap='Greys_r')
#plt.show()
################################################################################
## analyze randomized z with the estimated Hs
# goes to figure in 1.3 in the report
rrs_v = np.array(rrs)
Hs_v = np.array(Hs)
plt.close()
plt.hold(False)
#plt.scatter(rrs_v[:,2],rrs_v[:,0], c=Hs_v, linewidths=0.1,s=10)

f,ax = plt.subplots(3,3)
k=0
v = np.linspace(0,1,20)
for i in range(ax.shape[0]):
    for j in range(ax.shape[1]):

        if k<rrs_v.shape[1]:
            ax[i,j].scatter(rrs_v[:,k],Hs_v, linewidths=0.1,s=10)
            ax[i,j].hold(True)
            p0, p1, rval, _, _ = linregress(rrs_v[:,k],Hs_v)
            ax[i,j].plot(v,p0*v+p1,'r',linewidth=3)
            ax[i,j].set_title('r^2=%f'%rval**2)
        k+=1


## plot the model!
plot_model(vae,'model1_vae.png')
plot_model(encoder,'model1_encoder.png')
plot_model(generator,'model1_generator.png')

##
import sys
reload(sys.modules['keras'])
