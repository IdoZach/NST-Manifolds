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
from sklearn.model_selection import train_test_split

from scipy.stats import norm
from scipy.stats import linregress
from keras.layers import Input, Dense, Lambda, Convolution2D, \
    Conv2D, Flatten, Conv2DTranspose, RepeatVector, LSTM # there's also convlstm2d
from keras.models import Model
from keras import backend as K
from keras import metrics
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

def gauss2D_filter(shape=(9,9),sigma=0.5):
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
    h = np.sum(h,axis=0) # turn to 1D
    return h
## ################################################
## generate fbms
reCalc = True
#reCalc = False
n=256
N=10000
if os.path.exists('data.bin') and not reCalc:
    print 'loading data from file'
    Xtrain, Xtest, Ytrain, Ytest = pickle.load(open('data.bin','r'))
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

    pickle.dump([Xtrain,Xtest,Ytrain,Ytest],open('data.bin','w'))

## ################################################
## train and learn network

batch_size = 100
original_dim = n
latent_dim = 7
intermediate_dim = n/2
intermediate_dim2 = n/4
intermediate_dim3 = n/8
epochs = 20
epsilon_std = 1

x = Input(shape=(original_dim,))

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

y = Lambda(scale_space)(x)
print 'y',y
yr = LSTM(units=n)(y)
print 'yr',yr
h = Dense(intermediate_dim, activation='relu')(yr)
h_t=h
h_b=h
"""
h_expand = Lambda( lambda x: K.expand_dims(K.expand_dims(x,1),3)) (h)
print 'h_expand',h_expand
#h_expand = K.expand_dims(h_expand,3)
n_filters = 15
ker_dim = 7
h_t = Convolution2D(n_filters,[1,ker_dim],activation='tanh')(h_expand)
h_b = Convolution2D(n_filters,[1,ker_dim],activation='tanh')(h_expand)
conv_out_dim = n_filters * (intermediate_dim-ker_dim+1)
h_t = Flatten()(h_t)
h_b = Flatten()(h_b)
"""
def normgrad(x):
    gradfilt = np.array([1,-1],dtype=np.float32)
    gradfilt = np.expand_dims(gradfilt,1)
    gradfilt = np.expand_dims(gradfilt,2)
    res = tf.nn.conv1d(K.expand_dims(x,2),gradfilt,stride=1,padding='VALID')
    res = K.squeeze(res,2)
    #print K.int_shape(res)
    return K.log(K.square(res))
    #print K.int_shape(res)
    #return res


h_top = Dense(intermediate_dim2, activation = 'relu')(h_t)
#h_top2 = Dense(intermediate_dim3, activation = 'tanh')(h_top)
h_bottom = Dense(intermediate_dim2, activation = 'relu')(h_b)
#h_bottom2 = Dense(intermediate_dim3, activation = 'tanh')(h_bottom)
#calc_logvar = Lambda(normgrad)(h)
#z_log_var = K.concatenate([calc_logvar, h])
z_mean = Dense(latent_dim)(h_top)
z_log_var = Dense(latent_dim)(h_bottom)

print 'z', z_mean, z_log_var

print 'built layers'
def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0.,
                              stddev=epsilon_std)
    return z_mean + K.exp(z_log_var / 2) * epsilon
    #z_log_var = K.maximum(0.01,z_log_var)
    #return z_mean + z_log_var / 2 * epsilon

# note that "output_shape" isn't necessary with the TensorFlow backend
z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

# we instantiate these layers separately so as to reuse them later
#decoder_h1 = Dense(intermediate_dim2, activation = 'tanh')
"""
decoder_h1 = Conv2DTranspose(5,(1,3), activation = 'tanh')
decoder_h2 = Conv2DTranspose(5,(1,5), activation = 'tanh')
decoder_h3 = Conv2DTranspose(5,(1,7), activation = 'tanh')
#h_decoded1 = decoder_h1(z)
#print K.shape(h_decoded1)
decoder_h3a = Flatten()
decoder_h4 = Dense(intermediate_dim, activation='tanh')
decoder_mean = Dense(original_dim, activation='tanh')
def decode_pipeline(z):
    zz = Lambda( lambda x: K.expand_dims(K.expand_dims(x,1),2) )(z)
    dec = decoder_h4(decoder_h3a(decoder_h3(decoder_h2(decoder_h1(zz)))))
    return decoder_mean(dec)
x_decoded_mean = decode_pipeline(z)#Lambda(decode_pipeline)(z)
"""
decoder_h2 = Dense(intermediate_dim2, activation='tanh')
decoder_h1= Dense(intermediate_dim, activation='tanh')
decoder_log_std = Dense(original_dim, activation='tanh')
decoder_mean = Dense(original_dim, activation='tanh')
decoder_ss = Lambda(scale_space)
decoder_rnn = LSTM(units=n)

def decode_pipeline(z):
    #zz = Lambda( lambda x: K.expand_dims(K.expand_dims(x,1),2) )(z)
    dec_mean = decoder_mean(decoder_h1(decoder_h2(z)))
    dec_log_std = decoder_log_std(decoder_h1(decoder_h2(z)))
    dec_mean = decoder_rnn(decoder_ss(dec_mean))
    return dec_mean, dec_log_std
#print 'decoded', x_decoded_mean
x_decoded_mean, x_decoded_log_std = decode_pipeline(z)
def vae_loss(x, x_decoded_mean):
    #x_decoded_mean, x_decoded_log_std = x_dec
    #xent_loss = original_dim * metrics.binary_crossentropy(x, x_decoded_mean)
    #print x_decoded_mean
    #print x_decoded_log_std
    #term1 = K.square(x-x_decoded_mean)/(2.0 * K.exp(x_decoded_log_std))
    #term2 = 0.5 * K.sum(x_decoded_log_std)
    #reconstruction_loss = original_dim*(term1 + term2)
    reconstruction_loss  = original_dim * metrics.mean_squared_error(x, x_decoded_mean)
    kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    #kl_loss = - 0.5 * K.sum(1 + K.log(0.1+z_log_var) - K.square(z_mean) - z_log_var, axis=-1)
    #return xent_loss + kl_loss
    return reconstruction_loss + kl_loss

vae = Model(x, outputs=x_decoded_mean)
vae.compile(optimizer='RMSprop', loss=vae_loss) # was RMSprop

# train the VAE on MNIST digits

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
#_z_expanded = Lambda( lambda x: K.expand_dims(K.expand_dims(x,1),2) )(decoder_input)
_x_decoded_mean, _x_decoded_log_std = decode_pipeline(decoder_input)
#_h_decoded = decoder_h2(decoder_h1(decoder_input))
#_x_decoded_mean = decoder_mean(_h_decoded)
generator = Model(decoder_input, _x_decoded_mean)

# display a 2D manifold of the digits
m = 16  # figure with 15x15 digits
# linearly spaced coordinates on the unit square were transformed through the inverse CDF (ppf) of the Gaussian
# to produce values of the latent variables z, since the prior of the latent space is Gaussian
grid_x = norm.ppf(np.linspace(0.05, 0.95, m))
grid_y = norm.ppf(np.linspace(0.05, 0.95, m))
grid_z = norm.ppf(np.linspace(0.05, 0.95, m))
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
    rr = np.random.random((latent_dim))
    rrs.append(rr)
    #rr[0]=0
    #z_sample = np.array([[xi, yi, zi, rr[0], rr[1]]])
    z_sample = np.array([rr])
    x_decoded = generator.predict(z_sample)
    #digit = x_decoded[0].reshape(digit_size, digit_size)
    H_est, rsquare = hurst(x_decoded[0])
    Hs.append(H_est)
    rsquares+=rsquare
    tot+=1
    if not i % 20:
        spl[0].hold(False)
        diffres = x_decoded[0][1:]-x_decoded[0][:-1]
        _, bins, patches = spl[0].hist(diffres, 50, normed=1, facecolor='green', alpha=0.75)
        y = mlab.normpdf( bins, np.mean(diffres), np.std(diffres))
        spl[0].hold(True)
        spl[0].plot(bins, y, 'r--', linewidth=1)
        spl[0].set_title('diff, exp %d/%d'%(i,num_exps))
        spl[1].hold(False)
        spl[1].plot(x_decoded[0])
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
