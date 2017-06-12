'''This script demonstrates how to build a variational autoencoder with Keras.
Reference: "Auto-Encoding Variational Bayes" https://arxiv.org/abs/1312.6114
'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import kurtosis
from scipy.misc import imsave
from scipy.misc import imresize

from keras.layers import Input, Dense, Lambda, Layer, Activation
from keras.layers import ConvLSTM2D, Conv2D, Conv2DTranspose, Flatten, Reshape, merge, Merge, MaxPool2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import Concatenate
from tensorflow.contrib.keras import layers as tf_lay
from keras.models import Model
from keras import backend as K
from keras import metrics
from keras.datasets import mnist
from keras.optimizers import RMSprop

import tensorflow as tf
import fbm_data
from fbm_data import generate_2d_fbms, get_kth_imgs
from local_phase_model import LocalPhase

# train the VAE on MNIST digits
original_dim = 32*32# 784
original_dim2 = int(np.sqrt(original_dim))
(x_train0, y_train0), (x_test0, y_test0) = mnist.load_data()
n=32
use_fbms = False
batch_size = 100
if use_fbms:
    print('USING fBms')
    x_train0, x_test0, y_train0, y_test0 = \
        generate_2d_fbms(N=50000,n=n,reCalc=False,resize=original_dim)
else:
    print('using real imgs')
    x_train0, x_test0, y_train0, y_test0 = \
        get_kth_imgs(N=50000,n=n,reCalc=False,resize=original_dim)
    localphase = LocalPhase(x_train0, x_test0, y_train0, y_test0, recalc_stats=True)
    y_test0 = localphase.agg_y_test
    y_train0 = localphase.agg_y_train
    maxlen = (len(x_train0)/batch_size) * batch_size
    x_train0 = x_train0[:maxlen]
    y_train0 = y_train0[:maxlen]
    maxlen = (len(x_test0)/batch_size) * batch_size
    x_test0 = x_test0[:maxlen]
    y_test0 = y_test0[:maxlen]
############################################################


latent_dim = 5
latent_dim_w = 64
intermediate_dim = 256
intermediate_dim2 = 256
intermediate_dim3 = 128
epochs = 20
epsilon_std = 1.0


x = Input(batch_shape=(batch_size, original_dim))
noise = Input(batch_shape=(batch_size, latent_dim_w))#np.random.randn(batch_size,latent_dim_w)
x_cond = merge([x,noise],mode='concat',concat_axis=1)
use_my = True

# ------- my implementation ---------------------------------------------
# -------------------------------------------------------------------------
conv_nfilters = 10
conv_filtsize = 3
rnn_nfilters = 10
rnn_filtsize = 5
rec_iters = 4
iter_conv_nfilters = 10
iter_conv_filtsize = 3
def my_encoder(x):
    x_in = Reshape((original_dim2,original_dim2))(x)
    x_in = Lambda( lambda x: K.expand_dims(x,3))(x_in)
    init_conv_res = Conv2D(conv_nfilters,conv_filtsize,activation='relu',padding='same')(x_in)
    iter_conv = Conv2D(iter_conv_nfilters,iter_conv_filtsize,activation='relu',padding='same')
    iter_conv2 = Conv2D(iter_conv_nfilters,iter_conv_filtsize,activation='relu',padding='same')
    conv_res = [ iter_conv(init_conv_res) ]
    for i in range(rec_iters-1):
        this_res = iter_conv2(conv_res[-1])
        conv_res.append(this_res)
    conv_res2=[]
    for i in conv_res:
        conv_res2.append( Lambda(lambda x : K.expand_dims(x,axis=1) )(i) )
    conv_res = Concatenate(axis=1)(conv_res2)
    rnn_unit = ConvLSTM2D(rnn_nfilters,rnn_filtsize,activation='relu')
    #conv_res = Lambda( lambda x: K.expand_dims(x,1))(conv_res) # time steps in 1st dimension
    #print conv_res
    rnn_res = rnn_unit(conv_res)
    #print(conv_res)
    #print(rnn_res)
    #for i in range(rec_iters-1):
    #    rnn_res = Lambda( lambda x: K.expand_dims(x,1))(rnn_res)
    #    rnn_res = rnn_unit(rnn_res) # what about the state?
    w_flat = Flatten()(rnn_res)
    z_flat = Flatten()(conv_res2[-1])
    z_mean = Dense(latent_dim)(z_flat)
    z_log_var = Dense(latent_dim)(z_flat)

    w_mean = Dense(latent_dim_w)(w_flat)
    w_log_var = Dense(latent_dim_w)(w_flat)

    return z_mean, z_log_var, w_mean, w_log_var

def my_decoder(z,w,units=None):
    if units is None:
        z_dense = Dense(original_dim,activation='relu')
        w_dense = Dense(original_dim,activation='relu')

        rnn_input = Dense(original_dim)
        rnn_unit = ConvLSTM2D(rnn_nfilters,rnn_filtsize,padding='same',activation='relu')
        iter_conv2 = Conv2DTranspose(iter_conv_nfilters,iter_conv_filtsize,activation='relu',padding='same')
        iter_conv = Conv2DTranspose(iter_conv_nfilters,iter_conv_filtsize,activation='relu',padding='same')
        #last_deconv = Conv2DTranspose(conv_nfilters,conv_filtsize, activation='relu')
        last_conv = Conv2D(1,5,activation='relu',padding='same')

        units = [z_dense, w_dense, rnn_input, rnn_unit, iter_conv, iter_conv2, last_conv]

    z_dense, w_dense, rnn_input, rnn_unit, iter_conv, iter_conv2, last_conv = units

    #z_out = z_dense(z)
    w_out = w_dense(w)
    #concatted = Concatenate(axis=-1)([z_out,w_out])

    conv_input_2d = Reshape((original_dim2,original_dim2))(w_out)
    conv_input_2d = Lambda(lambda x: K.expand_dims(x,3))(conv_input_2d)
    conv_res = [ iter_conv(conv_input_2d) ]
    for i in range(rec_iters-1):
        this_res = iter_conv2(conv_res[-1])
        conv_res.append(this_res)
    conv_res2=[]
    for i in conv_res:
        conv_res2.append( Lambda(lambda x : K.expand_dims(x,axis=1) )(i) )
    conv_res3 = Concatenate(axis=1)(conv_res2)

    #rnn_res = rnn_unit(conv_res3)
    # now its with rnn_nfilters channels, need to flatten in the depth channel
    #rnn_res = last_conv(rnn_res)

    rnn_res = last_conv(conv_res[-1])
    #print('rnn res',rnn_res)

    #decoded = last_deconv(rnn_res)
    decoded = Flatten()(rnn_res)

    return decoded, units


# -------------------------------------------------------------------------


# -------------------------------------------------------------------------
# --------end my implementation--------------------------------------------

# ----------- att2image-type implementation -------------------------------

my2_conv_channels = [64, 128, 256, 256]
my2_conv_filts = [5, 5, 3, 4]
my2_prepad = [True, True, True, False]
my2_dense = [256]
def cv2_encoder(x, noise):

    #h = Dense(intermediate_dim, activation='relu')(x)
    #x_in1 = Concatenate()([x, noise])
    #x_in = Dense(original_dim)(x_in1)
    #x_in = Reshape((original_dim2,original_dim2))(x_in)
    x_in = Reshape((original_dim2,original_dim2))(x)
    x_ = Lambda( lambda x: K.expand_dims(x,3))(x_in)

    # create network
    enc_net = []
    for chan,filt,pad in zip(my2_conv_channels, my2_conv_filts, my2_prepad):
        if pad:
            enc_net.append(ZeroPadding2D((filt/2,filt/2)))
        enc_net.append(Conv2D(chan,filt))
        #enc_net.append(BatchNormalization())
        enc_net.append(Activation('relu'))
        if pad:
            enc_net.append(MaxPool2D())
    enc_net.append(Flatten())
    for sz in my2_dense:
        enc_net.append(Dense(sz))

    # encode
    print('enc net',enc_net)
    for lay in enc_net:
        x_ = lay(x_)
    # x_ is now of size 1024

    z_mean = Dense(latent_dim,activation='relu')(x_)
    z_log_var = Dense(latent_dim,activation='relu')(x_)

    #w_mean = Dense(latent_dim_w)(h)
    #w_log_var = Dense(latent_dim_w)(h)

    return z_mean, z_log_var# , w_mean, w_log_var

def cv2_decoder(z_cond,units=None):
    #dd = original_dim2 - fsize + 1
    #n_filters = 5
    if units is None:
        dec_net = []
        dec_net.append(Dense(256,activation='relu'))
        for sz in my2_dense[::-1]:
            dec_net.append(Dense(sz,activation='relu'))
        dec_net.append(Reshape((1,1,my2_dense[-1])))
        # 1 x 1 x 256
        for chan,filt,pad in zip(my2_conv_channels[::-1], my2_conv_filts[::-1], my2_prepad[::-1]):
            strides = (1,1)#(1,1) if pad else (1,1)
            dec_net.append(Conv2DTranspose(chan,filt,strides=strides))
            #dec_net.append(BatchNormalization())
            dec_net.append(Activation('relu'))
        dec_net.append(Conv2DTranspose(1,6,strides=(2,2),activation='relu'))
        dec_net.append(Flatten())

        units = dec_net

    #d_ = Dense(256,activation='relu')(z_cond)
    d_ = Lambda(lambda x: x)(z_cond) # identity...
    print('UNITS',units)
    for lay in units:
        d_ = lay(d_)

    print('decoded',d_)
    return d_, units

# -------- cvae implementation --------------------------------------------

def cv_encoder(x_cond):

    h = Dense(intermediate_dim, activation='relu')(x)
    #h = Reshape((16,16))(h)
    #h = Lambda(lambda x: K.expand_dims(x,3))(h)
    #h1 = Conv2D(10,5,strides=(2,2),padding='valid',activation='relu')(h)
    #h1 = Flatten()(h1)
    h1=Dense(intermediate_dim2, activation='tanh')(h)
    z_mean = Dense(latent_dim)(h1)
    z_log_var = Dense(latent_dim)(h1)

    #w_mean = Dense(latent_dim_w)(h)
    #w_log_var = Dense(latent_dim_w)(h)

    return z_mean, z_log_var# , w_mean, w_log_var

def cv_decoder(z_cond,units=None):
    fsize = 5
    fsize1 = 3
    dd = original_dim2 - fsize + 1
    n_filters = 5
    if units is None:
        decoder_h = Dense(intermediate_dim, activation='relu')
        decoder_mean = Dense(original_dim, activation='tanh')

        w_decoder = Dense(n_filters * dd * dd, activation='tanh')
        conv_dec1 = Conv2DTranspose( n_filters,fsize1,strides=(2,2),padding='valid',activation='relu')
        conv_dec2 = Conv2DTranspose( n_filters,fsize1,strides=(2,2),padding='valid',activation='relu')
        conv_dec = Conv2DTranspose( n_filters,fsize,strides=(1,1),padding='valid',activation='tanh')
        #w_decoder_mean = Dense(original_dim, activation='tanh')

        units = [decoder_h, decoder_mean, w_decoder, conv_dec, conv_dec1, conv_dec2]#, w_decoder_mean]

    #decoder_h, decoder_mean, w_decoder, w_decoder_mean = units
    decoder_h, decoder_mean, decoder_w, conv_dec, conv_dec1, conv_dec2= units
    h_decoded = decoder_h(z_cond)
    w_decoded = decoder_w(h_decoded)
    reshaped = Reshape((dd,dd,n_filters))(w_decoded)
    deconved = conv_dec1(reshaped)
    deconved = conv_dec2(deconved)
    deconved = conv_dec(deconved)
    deconved = Flatten()(deconved)
    x_decoded_mean = decoder_mean(deconved)
    #w_decoded = w_decoder(w)
    #w_decoded_mean = w_decoder_mean(w_decoded)
    #x_decoded_mean = Lambda(lambda x: 0.5*(x+w_decoded_mean))(x_decoded_mean)

    return x_decoded_mean, units

# --------- end cvae implementation ---------------------------------------

def default_encoder(x):
    h = Dense(intermediate_dim, activation='relu')(x)
    z_mean = Dense(latent_dim)(h)
    z_log_var = Dense(latent_dim)(h)

    w_mean = Dense(latent_dim_w)(h)
    w_log_var = Dense(latent_dim_w)(h)

    return z_mean, z_log_var, w_mean, w_log_var

def default_decoder(z,w,units=None):
    if units is None:
        decoder_h = Dense(intermediate_dim, activation='relu')
        decoder_mean = Dense(original_dim, activation='tanh')
        w_decoder = Dense(intermediate_dim, activation='relu')
        w_decoder_mean = Dense(original_dim, activation='tanh')

        units = [decoder_h, decoder_mean, w_decoder, w_decoder_mean]

    decoder_h, decoder_mean, w_decoder, w_decoder_mean = units
    h_decoded = decoder_h(z)
    x_decoded_mean = decoder_mean(h_decoded)
    w_decoded = w_decoder(w)
    w_decoded_mean = w_decoder_mean(w_decoded)
    x_decoded_mean = Lambda(lambda x: 0.5*(x+w_decoded_mean))(x_decoded_mean)

    return x_decoded_mean, units


use_encoder = cv2_encoder if use_my else default_encoder
use_decoder = cv2_decoder if use_my else default_decoder

#z_mean, z_log_var, w_mean, w_log_var = use_encoder(x,noise)
z_mean, z_log_var = use_encoder(x,noise)

def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0.,
                              stddev=epsilon_std)
    return z_mean + K.exp(z_log_var / 2) * epsilon

#def sampling_w(args):
#    w_mean, w_log_var = args
#    epsilon = K.random_normal(shape=(batch_size, latent_dim_w), mean=0.,
#                              stddev=epsilon_std)
#    return w_mean + K.exp(w_log_var / 2) * epsilon

# note that "output_shape" isn't necessary with the TensorFlow backend
z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])
z_cond = merge([z, noise],mode='concat',concat_axis=1)
#w = Lambda(sampling_w, output_shape=(latent_dim_w,))([w_mean, w_log_var])

# we instantiate these layers separately so as to reuse them later

x_decoded_mean, units = use_decoder(z_cond)

# Custom loss layer
class CustomVariationalLayer(Layer):
    def __init__(self, **kwargs):
        self.is_placeholder = True
        super(CustomVariationalLayer, self).__init__(**kwargs)

    def vae_loss(self, x, x_decoded_mean):
        #xent_loss = original_dim * metrics.binary_crossentropy(x, x_decoded_mean)
        xent_loss = original_dim * metrics.mean_squared_error(x, x_decoded_mean)
        kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        #kl_loss_w = - 0.5 * K.sum(1 + w_log_var - K.square(w_mean) - K.exp(w_log_var), axis=-1)
        return K.mean(xent_loss + kl_loss)# + kl_loss_w)

    def vae_ssim_loss(self, x,x_decoded_mean):
        #xent_loss = original_dim * metrics.binary_crossentropy(x, x_decoded_mean)
        dim1=int(np.sqrt(original_dim))
        x = K.reshape(x,[-1,dim1,dim1])
        x_decoded_mean = K.reshape(x_decoded_mean,[-1,dim1,dim1])

        ssim_loss = original_dim * fbm_data.loss_DSSIS_tf11(x,x_decoded_mean, False)#,int(np.sqrt(original_dim)))

        kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        #kl_loss_w = - 0.5 * K.sum(1 + w_log_var - K.square(w_mean) - K.exp(w_log_var), axis=-1)
        #ssim_loss = tf.Print(ssim_loss,[K.get_variable_shape(xent_loss+kl_loss), K.get_variable_shape(kl_loss)],'xent ssim ')
        #kl_loss_w = - 0.5 * K.sum(1 + w_log_var - K.square(w_mean) - K.exp(w_log_var), axis=-1)
        #print xent_loss, ssim_loss, kl_loss
        return K.mean(ssim_loss + kl_loss)

    def call(self, inputs):
        x = inputs[0]
        x_decoded_mean = inputs[1]
        loss = self.vae_loss(x, x_decoded_mean)
        #loss = self.vae_cos_loss(x, x_decoded_mean)
        #loss = self.vae_frac_loss(x, x_decoded_mean)
        #loss = self.vae_ssim_loss(x, x_decoded_mean)
        self.add_loss(loss, inputs=inputs)
        # We won't actually use the output.
        return x

def vae_loss(x_in, x_out):
    x = x_in[0]
    x_decoded_mean = x_out

    xent_loss = original_dim * metrics.binary_crossentropy(x, x_decoded_mean)
    kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    #kl_loss_w = - 0.5 * K.sum(1 + w_log_var - K.square(w_mean) - K.exp(w_log_var), axis=-1)
    #kl_loss_w = - 0.5 * K.sum(1 + w_log_var - K.square(w_mean) - K.exp(w_log_var), axis=-1)
    #kl_loss = tf.Print(kl_loss,
    #                   [K.get_variable_shape(xent_loss)],
    #                   'xent ssim ')
    return K.mean(xent_loss + kl_loss)# + kl_loss_w)
y = CustomVariationalLayer()([x, x_decoded_mean])
#vae = Model([x, noise], x)
vae = Model([x, noise], y)
rmsprop = RMSprop()
vae.compile(optimizer='rmsprop', loss=None)#vae_loss)


x_train = x_train0.astype('float32') / 128.
x_test = x_test0.astype('float32') / 128.
#x_train = x_train - 1
#x_test = x_test - 1
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
y_train = y_train0
y_test = y_test0

noise_train = np.random.randn(len(x_train0),latent_dim_w)
noise_test =  np.random.randn(len(x_test0),latent_dim_w)

#x_train = [ [x1,v1] for x1,v1 in zip(x_train,noise_train) ]
#x_test = [ [x1,v1] for x1,v1 in zip(x_test,noise_test) ]
x_train_c = [x_train, noise_train]
x_test_c = [x_test, noise_test]
vae.fit(x_train_c,
        shuffle=True,
        epochs=epochs,
        verbose=2,
        batch_size=batch_size,
        validation_data=(x_test_c, x_test))

# build a model to project inputs on the latent space
encoder = Model([x, noise], z_mean)

## ( todo separator ) display a 2D plot of the digit classes in the latent space
x_test_encoded = encoder.predict([x_test, noise_test], batch_size=batch_size)
plt.figure(figsize=(6, 6))
# first item in y is H, second is kurtosis
# next 10 are phase related
z = np.polyfit(range(10),np.log(1+y_test[:,2:]).T,deg=2)
latdisp = [0, 2]
plt.scatter(x_test_encoded[:, latdisp[0]], x_test_encoded[:, latdisp[1]], c=z[2,:], lw=0, s=8)

#plt.scatter(x_test_encoded[:, 0], z[2,:], lw=0, s=8)

#plt.scatter(x_test_encoded[:, latdisp[0]], x_test_encoded[:, latdisp[1]], c=np.mean(y_test[:,2:4],axis=1),lw=0,s=8)
#plt.scatter(x_test_encoded[:, latdisp[0]], x_test_encoded[:, latdisp[1]], c=np.log(y_test[:,1]),lw=0,s=8)

#plt.scatter(x_test_encoded[:, latdisp[0]], x_test_encoded[:, latdisp[1]], c=y_test[:,1],lw=0,s=8)

#plt.scatter(x_test_encoded[:,3], y_test,lw=0,s=8)
plt.xlabel('Latent variable %d'%latdisp[0])
plt.ylabel('Latent variable %d'%latdisp[1])
plt.colorbar()
plt.show()
#plt.savefig('exp_fbm_lat.pdf')
## todo separation line

# build a digit generator that can sample from the learned distribution
decoder_input = Input(shape=(latent_dim,))
decoder_input_w = Input(shape=(latent_dim_w,))
#_h_decoded = decoder_h(decoder_input)
#_x_decoded_mean = decoder_mean(_h_decoded)
_z_cond = merge([decoder_input, decoder_input_w],mode='concat',concat_axis=1)
_x_decoded_mean, _ = use_decoder(_z_cond,units)
generator = Model([decoder_input, decoder_input_w], _x_decoded_mean)

# display a 2D manifold of the digits
n = 10  # figure with 15x15 digits
digit_size = int(np.sqrt(original_dim))
figure = np.zeros((digit_size * n, digit_size * n))
# linearly spaced coordinates on the unit square were transformed through the inverse CDF (ppf) of the Gaussian
# to produce values of the latent variables z, since the prior of the latent space is Gaussian
h_figure = np.zeros((n,n,3))
grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
grid_y = norm.ppf(np.linspace(0.05, 0.95, n))

figure2 = np.zeros((digit_size * n, digit_size * n))
example_imgs = np.zeros((digit_size, digit_size,6))
example_Hs_ind = 0
example_imgs_stats = np.zeros((6,3))
def mat2gray(x):
    x = x - np.min(x)
    x = x / np.max(x)
    return x

for i, yi in enumerate(grid_x):
    for j, xi in enumerate(grid_y):
        mani_pos = [0,1]
        z_sample = np.random.randn(1,latent_dim)
        z_sample[:,mani_pos] = [xi,yi]
        #z_sample = np.array(np.hstack([ [[xi, yi]],np.random.randn(1,latent_dim-2) ]))*1
        w_sample = np.random.randn(1,latent_dim_w)*1
        # randomize w instead of z as the manifold parameters
        # here we should see a disoriented behaviour
        #z_sample = np.random.randn(1,latent_dim)*1
        #w_sample = np.concatenate([np.array([[xi, yi]]), [np.random.randn(latent_dim_w-2)*1]],axis=1)

        x_decoded = generator.predict([z_sample,w_sample])
        digit = x_decoded[0].reshape(digit_size, digit_size)
        h_est = fbm_data.hurst2d(digit,max_tau=5)
        grad = np.gradient(digit)
        kurt = kurtosis(grad[0][2:-1,2:-1].flatten())
        #kurt = kurtosis(grad[0][2:-1,2:-1].flatten())
        hh, r2 = h_est
        h_figure[i,j,:] = np.array([hh, r2, kurt])

        hh_entry = int(hh*10)-3
        if hh_entry>=0 and hh_entry<6:
            example_imgs[:,:,hh_entry] = mat2gray(digit)
            example_imgs_stats[hh_entry,:] = [hh, r2, kurt]


        if hh<0.5:# and False:
            plt.hold(False)
            plt.imshow(digit,interpolation='none')
            plt.title('H=%f, R^2=%f'%(hh,r2))
            plt.show()
            plt.pause(0.2)
        #raw_input('bla')


        figure[i * digit_size: (i + 1) * digit_size,
               j * digit_size: (j + 1) * digit_size] = mat2gray(digit)

        # get also random images from the dataset

        rr = np.random.randint(0,x_train.shape[0])
        digit2 = np.reshape(x_train[rr,:],[original_dim2,original_dim2])
        figure2[i * digit_size: (i + 1) * digit_size,
               j * digit_size: (j + 1) * digit_size] = mat2gray(digit2)


#plt.figure(figsize=(10, 10))
figure=figure-np.min(figure)
figure=figure/np.max(figure)
_,pp=plt.subplots(2,2)
pp[0,0].imshow(figure, cmap='gray',interpolation='none')
#pp[1,1].imshow(figure2, cmap='gray',interpolation='none')
#.show()
## (todo separator) display several examples
example_im = np.vstack(np.transpose(example_imgs,[2,0,1])).T
plt.imshow(example_im,interpolation='none',cmap='gray')
print example_imgs_stats
imsave('syn_fbms.png',imresize(example_im,example_im.shape*4,interp='nearest'))


## ( todo separator ) display h values
plt.figure(1)
plt.clf()
plt.imshow(h_figure[:,:,0],cmap='gray',interpolation='none',
           extent= [ grid_x[0], grid_x[-1] ,grid_y[0], grid_y[-1]] )
plt.xlabel('Latent variable 0')
plt.ylabel('Latent variable 1')
plt.colorbar()
plt.colorbar(res,ax=pp[0,1])
plt.savefig('exp_fbm_syn_h.pdf')
## ( todo separator )
res=pp[1,0].imshow(h_figure[:,:,1],cmap='gray',interpolation='none')
plt.colorbar(res,ax=pp[1,0])
pp[1,0].set_title('R^2')
res=pp[1,1].imshow(h_figure[:,:,2],cmap='gray',interpolation='none')
plt.colorbar(res,ax=pp[1,1])
pp[1,1].set_title('Kurtosis')

## ( todo separator ) show scatter of K vs H
plt.figure(1,figsize=(8,5))
plt.hold(False)
plt.scatter(h_figure[:,:,0].flatten(),
            h_figure[:,:,2].flatten(),
            #c=h_figure[:,:,1].flatten() ,
            lw=0,s=8)
plt.xlabel('H')
plt.ylabel('Kurtosis')
plt.savefig('exp_fbm_syn1.pdf')
#plt.colorbar(format='%1.2e')
plt.show()
