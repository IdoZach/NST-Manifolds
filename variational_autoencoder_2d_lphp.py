'''This script demonstrates how to build a variational autoencoder with Keras.

Reference: "Auto-Encoding Variational Bayes" https://arxiv.org/abs/1312.6114
'''
import tsne
import os
from os import listdir
import cPickle as pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import mlab
from fbm.fbm import fbm
from fbm2d import synth2, hurst2d
from sklearn.model_selection import train_test_split

from scipy.stats import norm, kurtosis
from scipy.stats import linregress
from PIL import Image

import tensorflow as tf
from keras.layers import Input, Dense, Lambda, MaxPool2D, \
    Conv2D, Flatten, Conv2DTranspose, RepeatVector, LSTM, \
    Reshape, Merge, merge # there's also convlstm2d
from keras.models import Model
from keras import backend as K
from keras import metrics
from keras.utils import plot_model
from keras.datasets import mnist
from keras.layers import concatenate
from keras import callbacks

from fbm_data import generate_2d_fbms
# https://gist.github.com/Dref360/a48feaecfdb9e0609c6a02590fd1f91b

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


## load data
n=16
Xtrain, Xtest, Ytrain, Ytest = generate_2d_fbms(N=50000,n=n,reCalc=False)
#Xtrain, Xtest, Ytrain, Ytest = get_kth_imgs(N=50000,n=n,reCalc=False)
#Ytrain = np.array(Ytrain)
#Ytest = np.array(Ytest)

## ################################################
# train and learn network

batch_size = 200
# concat to match whole multiplication of batch size
max_data_sz = ( len(Xtrain) / batch_size ) * batch_size
Xtrain = Xtrain[:max_data_sz]
Ytrain = Ytrain[:max_data_sz]
max_data_sz = ( len(Xtest) / batch_size ) * batch_size
Xtest = Xtest[:max_data_sz]
Ytest = Ytest[:max_data_sz]

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

def decode_pipeline_lp(z):
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
#print 'x...', x
#y_hp = Lambda(scale_space_2d)(x)
#print 'y_hp',y_hp

#shared_LSTM = LSTM(units=n_units,return_sequences=False)
noise_dim = 80
noise_dim2 = 10

Recur_unit = Dense(noise_dim-noise_dim2,activation='relu')
def recur_layer(unit,z,x):
    out1 = unit(x)
    out2 =  Dense(noise_dim2,activation='relu')(z)
    return concatenate([out1,out2],axis=1)
x_flat = Flatten()(x)
x_flat = Lambda(lambda x: x-K.min(x))(x_flat)
#merger = concatenate([x_flat, z_mean],axis=1)
#print merger
x_flat_compressed = Dense(noise_dim,activation=None)(x_flat)
yr1 = recur_layer(Recur_unit,z_mean,
                  recur_layer(Recur_unit,z_mean,
                              recur_layer(Recur_unit,z_mean,
                                          recur_layer(Recur_unit,z_mean,x_flat_compressed))))

#yr1 = Lambda(lambda x: x-K.mean(x))(yr1) ####
#yr1 = Lambda(lambda x: x*0)(yr1) # check if it does anything...

#yr = shared_LSTM(y_hp) # this should be the generating noise
#print 'yr',yr # now this should actually be some noise factor

#z2_mean = Dense(noise_dim,activation='relu')(yr1)
#z2_log_var = Dense(noise_dim,activation='relu')(yr1)
z2_mean = Lambda(lambda x: K.mean(x,axis=-1,keepdims=True))(yr1)
print 'blbalb',z2_mean, yr1
z2_log_var = Lambda(lambda x: K.log(K.std(x,axis=-1,keepdims=True)))(yr1)
z2_mean = Dense(noise_dim,activation=None)(z2_mean)
z2_log_var = Dense(noise_dim,activation=None)(z2_log_var)


def sampling_hp(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(batch_size, noise_dim),
                              mean=0., stddev=epsilon_std)
    res = z_mean + K.exp(z_log_var)*epsilon
    return res#z_mean + K.exp(z_log_var) * epsilon

print 'built layers'
z_noise = Lambda(sampling_hp, output_shape=(noise_dim,))([z2_mean, z2_log_var])
print 'z noise', z_noise
# decoding hp
#decoder_hp_rnn = shared_LSTM # LSTM(units=n_units)
dec_Recur_unit = Dense(n**2-noise_dim2,activation='relu') # for mean
#dec_Recur_unit2 = Dense(n**2,activation='relu') # for mean
dec_Recur_unit2 = Conv2D(1,
                         kernel_size=3,
                         padding='same',
                         strides=1,
                         activation='relu')
dec_units = [dec_Recur_unit, dec_Recur_unit2]



def dec_recur_layer(unit,z,x):
    z_fc =  Dense(noise_dim2,activation='relu')(z)
    out1 = unit(x)

    return concatenate([out1,z_fc],axis=1)

def dec_recur_layer2(units,z,x):
    z_fc = Dense(noise_dim2,activation='relu')(z)
    out1 = units[0](x)
    inp = concatenate([out1,z_fc],axis=1)
    #z_fc = units[1]()
    inp2d = Reshape((n,n))(inp)
    inp2d = K.expand_dims(inp2d,3)
    z_fc = units[1](inp2d)

    z_fc = K.reshape(z_fc,(batch_size,n**2))
    #z_fc = K.squeeze(z_fc,axis=2)

    return z_fc
#print 'decoder_rnn',decoder_rnn
#summation = Lambda(lambda x: K.sum(x,axis=1)) # not sure about the axis
decoder_hp_fc = Dense(original_dim**2)


def decode_pipeline_cnn(z, z_noise):
    int_dim = 10
    filt_dim = 3

    noise_w = tf.Variable(tf.random_normal([noise_dim,n**2]))
    z_noise_im = tf.reshape(tf.matmul(z_noise,noise_w),[-1,n,n])
    z_noise_im = tf.expand_dims(z_noise_im,3)
    z_noise_im = tf.transpose(z_noise_im,[3,1,2,0])

    dec_filt0 = tf.Variable(tf.random_normal([latent_dim,int_dim]))
    dec_filt0 = tf.nn.relu(tf.matmul(z,dec_filt0))
    def create_filt():
        lev1_weight = tf.Variable(tf.random_normal([int_dim,filt_dim**2]))
        dec_filt_lev1 = tf.matmul(dec_filt0,lev1_weight)
        dec_filt_lev1 = tf.reshape(dec_filt_lev1,[-1,filt_dim,filt_dim])
        #dec_filt_lev1 = tf.expand_dims(dec_filt_lev1,2)
        dec_filt_lev1 = tf.expand_dims(dec_filt_lev1,3)
        dec_filt_lev1 = tf.transpose(dec_filt_lev1,[1,2,0,3])# switch batch to last channel
        return dec_filt_lev1
    #print 'bla',z_noise_im,dec_filt_lev1
    dec_filt_lev1 = create_filt()
    dec_filt_lev2 = create_filt()
    dec_filt_lev3 = create_filt()
    dec_filt_lev4 = create_filt()
    dec_filt_lev5 = create_filt()

    filtered_lev1 = tf.nn.depthwise_conv2d(z_noise_im,dec_filt_lev1,[1,1,1,1],'SAME',name='conv1')
    filtered_lev1 = tf.nn.relu(filtered_lev1)
    filtered_lev2 = tf.nn.depthwise_conv2d(filtered_lev1,dec_filt_lev2,[1,1,1,1],'SAME',name='conv2')
    filtered_lev2 = tf.nn.relu(filtered_lev2)
    filtered_lev3 = tf.nn.depthwise_conv2d(filtered_lev2,dec_filt_lev3,[1,1,1,1],'SAME',name='conv3')
    filtered_lev3 = tf.nn.relu(filtered_lev3)
    filtered_lev4 = tf.nn.depthwise_conv2d(filtered_lev3,dec_filt_lev4,[1,1,1,1],'SAME',name='conv3')
    filtered_lev4 = tf.nn.relu(filtered_lev4)
    filtered_lev5 = tf.nn.depthwise_conv2d(filtered_lev4,dec_filt_lev5,[1,1,1,1],'SAME',name='conv3')
    filtered_lev5 = tf.nn.relu(filtered_lev5)

    res = tf.transpose(filtered_lev5,[3,1,2,0]) # turn batch back to the first dim
    res = tf.squeeze(res,3)

    # flatten
    #res = tf.reshape(res,[batch_size,n**2])
    print 'conv res',res
    return res

    #dec_filt1 = tf.nn.relu(dec_filt1)


def decode_pipeline_full(z, z_noise):
    x_decoded_lp_mean = decode_pipeline_lp(z)
    #print 'before'
    #print z_noise
    #dec = dec_recur_layer(dec_Recur_unit,z,
    #              dec_recur_layer(dec_Recur_unit,z,
    #                          dec_recur_layer(dec_Recur_unit,z,
    #                                      dec_recur_layer(dec_Recur_unit,z,z_noise))))
    """
    up_noise = Dense(n**2,activation=None)(z_noise)
    dec = dec_recur_layer2(dec_units,z,
        dec_recur_layer2(dec_units,z,
        dec_recur_layer2(dec_units,z,
        dec_recur_layer2(dec_units,z,up_noise))))
        #dec = Lambda(lambda x: x*0)(dec) ####
    #dec = Lambda(lambda x: x-K.mean(x))(dec) ####
    #dec = dec_Recur_unit(dec_Recur_unit(dec_Recur_unit(dec_Recur_unit(z_noise))))
    #dec = decoder_hp_rnn(z_noise)
    print 'after'

    dec = decoder_hp_fc(dec)
    #"""

    dec = Lambda( lambda zz : decode_pipeline_cnn(zz[0],zz[1]) )([z,z_noise])

    sum_outputs = False
    if sum_outputs:

        #flat_lp = Flatten()(x_decoded_lp_mean)
        #dec = concatenate([dec,flat_lp],axis=1)
        #dec = Dense(original_dim**2)(dec)
        dec = Reshape((n,n))(dec)
        scaler = Lambda(lambda x: K.ones_like(dec))(dec)
#        print 'merged',scaler
        merged = merge([x_decoded_lp_mean,scaler],mode='dot')
#        print 'merged',merged,dec,scaler
        #merged = Reshape((n,n))(merged)

        dec = Lambda(lambda x: x + merged )(dec)
    else: # join lp+hp outputs differently
        #dec = Lambda(lambda x: (x-K.mean(x))/K.std(x) )(dec)
        print 'x dec lp mean',x_decoded_lp_mean
        flat_lp = Flatten()(x_decoded_lp_mean)
        dec = Flatten()(dec)
        #flat_lp = K.reshape(x_decoded_lp_mean,[batch_size,n**2])
        #dec = K.reshape(dec,[-1,n**2])
        #flat_lp = Lambda(lambda x: (x-K.mean(x))/K.std(x) )(flat_lp)
        dec = concatenate([dec,flat_lp],axis=1)
        dec = Dense(original_dim**2)(dec)
        dec = Reshape((n,n))(dec)

        #dec = Lambda(lambda x: x + x_decoded_lp_mean)(dec)
    return dec

x_decoded_mean = decode_pipeline_full(z,z_noise)
#print 'decoded', x_decoded_mean
# # # # loss # # # # # # # # # # #######################################

def loss_logvarinc(y_true, y_pred):
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
    def lvi(y,f,trans=False):

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

        f1 = get1(f)[0] + get1(tf.transpose(f))[0]
        return f1/2.0#[0]

    lvi1_true = lambda _, f : lvi(y_true,f,trans=False)
    lvi1_pred = lambda _, f : lvi(y_pred,f,trans=False)
    dif = tf.constant([0,1,2,3])
    lvi_true = tf.scan(lvi1_true,dif,initializer=tf.constant(0.0))
    lvi_pred = tf.scan(lvi1_pred,dif,initializer=tf.constant(0.0))
    res = K.mean(K.square(lvi_true-lvi_pred))
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
    #y_true = tf.transpose(y_true, [0, 2, 3, 1])
    #y_pred = tf.transpose(y_pred, [0, 2, 3, 1])
    pad_method = 'VALID'
    patches_true = tf.extract_image_patches(y_true, [1, 10, 10, 1], [1, 1, 1, 1], [1, 1, 1, 1], pad_method)
    patches_pred = tf.extract_image_patches(y_pred, [1, 10, 10, 1], [1, 1, 1, 1], [1, 1, 1, 1], pad_method)

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
    ssim1 = K.mean(((1.0 - ssim) / 2))
    #ssim1 = tf.Print(ssim1,[ssim1],'SSIM')
    return ssim1


def vae_loss(x, x_decoded_mean):
    #x_decoded_mean, x_decoded_log_std = x_dec
    #xent_loss = original_dim * metrics.binary_crossentropy(x, x_decoded_mean)
    #print x_decoded_mean
    #print x_decoded_log_std
    #term1 = K.square(x-x_decoded_mean)/(2.0 * K.exp(x_decoded_log_std))
    #term2 = 0.5 * K.sum(x_decoded_log_std)
    #reconstruction_loss = original_dim*(term1 + term2)
    print 'X SHAPE',x
    use_mse = False
    if use_mse:
        term1 = K.flatten(x-x_decoded_mean)
        term1 =   K.mean(K.square(term1))
    else:
        #term1 = loss_DSSIS_tf11(x,x_decoded_mean,blur=True)
        term1 = loss_logvarinc(x,x_decoded_mean)
        #term1 = tf.Print(term1,[term1],'DSSIM')
    #reconstruction_loss  = original_dim**2 * metrics.mean_squared_error(x, x_decoded_mean)
    reconstruction_loss = original_dim**2 *term1
    print 'rec loss',reconstruction_loss

    kldiv_fun = lambda m1, lv1, m2, lv2 : \
        - 0.5 * K.sum(1 + lv1 - lv2 - (K.square(m1-m2)+K.exp(lv1))/np.exp(lv2), axis=-1)

    kl_loss = kldiv_fun(z_mean, z_log_var, 0, np.log(1))
    kl_loss2 = kldiv_fun(z2_mean, z2_log_var, 0, np.log(1))

    #kl_loss3 = kldiv_fun(x_decoded)
    """ some conditional entropy trials
    E_op = lambda x: 1/n**2 * K.sum(x,axis=0,keepdims=False)
    z2_invar = 1/(0.01+K.exp(z2_log_var))
    xf = K.reshape(x,[-1,n**2])
    xdecf = K.reshape(x_decoded_mean,[-1,n**2])
    print 'bef', z2_invar, xf
    Lx = z2_invar * xf
    print 'aft', Lx
    cond_ent1 = K.sum(xf * Lx,axis=-1)#K.dot(K.expand_dims(xf,axis=2),K.expand_dims(Lx,axis=1))
    cond_ent2 = -2*K.sum(xdecf * Lx,axis=-1)
    cond_ent3 = K.sum(xdecf * z2_invar * xdecf,axis=-1)
    print 'condent',cond_ent1
    cond_ent = - 0.5*K.sum(z2_log_var,axis=-1) + E_op(cond_ent1+cond_ent2+cond_ent3)
    print 'condent',cond_ent,'kl loss',kl_loss
    #"""
    #kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    #kl_loss2 = - 0.5 * K.sum(1 + z2_log_var - K.square(z2_mean) - K.exp(z2_log_var), axis=-1)
    #print 'KL LOSS2',kl_loss2
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
class Mycallback(callbacks.Callback):
    def on_epoch_begin(self, epoch, logs={}):
        mencoder = Model(x, [z_mean, z2_mean])
        # display a 2D plot of the digit classes in the latent space
        res = mencoder.predict(Xtest, batch_size=batch_size)
        print 'z  min %f, max %f, mean %f, std %f'%(np.min(res[0]),np.max(res[0]),np.mean(res[0]),np.std(res[0]))
        print 'z2 min %f, max %f, mean %f, std %f'%(np.min(res[1]),np.max(res[1]),np.mean(res[1]),np.std(res[1]))
mycallback = Mycallback()

vae.fit(Xtrain, Xtrain,
        shuffle=True,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(Xtest, Xtest), verbose=2,
        callbacks=[mycallback])
print 'done training.'
# build a model to project inputs on the latent space
encoder = Model(x, [z_mean, z2_mean])

# display a 2D plot of the digit classes in the latent space
x_test_encoded, x_test_encoded2 = encoder.predict(Xtest, batch_size=batch_size)

## get weights
for i, layer in enumerate(encoder.layers):
    if len(layer.get_weights()):
        i0 = layer.get_weights()[0]
        i1 = layer.get_weights()[1]
        print i, i0.shape, i1.shape
    if i==10 :#or i==10: # recursive unit
        print i0
        plt.imshow(i0,interpolation='none')
        plt.title('Rec unit')
        plt.show()

########################################################################################################################
# see the histograms of latent variables
plt.hold(True)
f, spl = plt.subplots(latent_dim)
for s,var in zip(spl,x_test_encoded.T):
    _, bins, patches = s.hist(var, 50, normed=1, facecolor='green', alpha=0.75)
    # add a 'best fit' line
    y = mlab.normpdf( bins, 0, 1)
    l = s.plot(bins, y, 'r--', linewidth=5)
## see histograms of random noise variables
N=5
_, spl = plt.subplots(N)
k=0
for i,var in enumerate(x_test_encoded2):
    spl[k].hold(False)
    res, bins, patches = s.hist(var, 50, normed=1, facecolor='green', alpha=0.75)
    # add a 'best fit' line

    y = mlab.normpdf( bins, np.mean(var), np.std(var))
    spl[k].plot(bins, y, 'r--', linewidth=5)
    spl[k].hold(True)
    spl[k].plot(bins[1:],res)


    spl[k].set_title(str(i))
    if k==4:
        plt.show()
        plt.pause(0.5)
        #raw_input()
    k=(k+1) % N

########################################################################################################################
#plt.figure(figsize=(6, 6))
# see correspondence between true latents and estimated
if x_test_encoded.shape[1]>=2:
    plt.clf(); plt.hold(False)
    def cart2pol(x, y):
        rho = np.sqrt(x**2 + y**2)
        phi = np.arctan2(y, x)
        return(rho, phi)
    polvec = cart2pol(x_test_encoded[:,1],x_test_encoded[:,2])
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
decoder_noise_input = Input(shape=(noise_dim,))
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
_, spl = plt.subplots(4)

plt.hold(True)
tot=0; rsquares=0
Hs = []
rrs = []
kurts = []
num_exps=1000
for i in range(num_exps):
    #z_sample = np.array([[xi, yi, zi]])
    rr = np.random.randn((latent_dim))*1
    #rr = [-0.1, 0.1, 0.1]
    rrs.append(rr)
    #rr[0]=0
    #z_sample = np.array([[xi, yi, zi, rr[0], rr[1]]])
    z_sample = np.array([rr])

    z_noise_sample = np.random.randn(noise_dim)
    z_noise_sample = np.array([z_noise_sample])*1
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
    diffres = np.squeeze(np.reshape(x_decoded[1:]-x_decoded[:-1],[1,-1]))
    kurt=kurtosis(diffres)
    kurts.append(kurt)
    mkurt = np.mean(np.array(kurts))

    rsquares+=rsquare
    tot+=1
    if not i % 30:
        spl[0].hold(False)


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
            spl[3].hold(False)
            spl[3].hist(np.array(kurts), 30, normed=1, facecolor='green', alpha=0.75)

        plt.title('H %f r2 %f mean r2 %f kurt %f mean %f '%(H_est,rsquare,rsquares/tot,kurt,mkurt))
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
