'''This script demonstrates how to build a variational autoencoder with Keras.
Reference: "Auto-Encoding Variational Bayes" https://arxiv.org/abs/1312.6114
'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from keras.layers import Input, Dense, Lambda, Layer
from keras.models import Model
from keras import backend as K
from keras import metrics
from keras.datasets import mnist

import tensorflow as tf
import fbm_data
from fbm_data import generate_2d_fbms


# train the VAE on MNIST digits
original_dim = 784
(x_train0, y_train0), (x_test0, y_test0) = mnist.load_data()
n=32
print 'USING fBms'
x_train0, x_test0, y_train0, y_test0 = \
    generate_2d_fbms(N=50000,n=n,reCalc=False,resize=original_dim)

############################################################

batch_size = 100
latent_dim = 2
latent_dim_w = 30
intermediate_dim = 256
epochs = 15
epsilon_std = 1.0


x = Input(batch_shape=(batch_size, original_dim))
h = Dense(intermediate_dim, activation='relu')(x)
z_mean = Dense(latent_dim)(h)
z_log_var = Dense(latent_dim)(h)

w_mean = Dense(latent_dim_w)(h)
w_log_var = Dense(latent_dim_w)(h)


def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0.,
                              stddev=epsilon_std)
    return z_mean + K.exp(z_log_var / 2) * epsilon

def sampling_w(args):
    w_mean, w_log_var = args
    epsilon = K.random_normal(shape=(batch_size, latent_dim_w), mean=0.,
                              stddev=epsilon_std)
    return w_mean + K.exp(w_log_var / 2) * epsilon

# note that "output_shape" isn't necessary with the TensorFlow backend
z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])
w = Lambda(sampling_w, output_shape=(latent_dim_w,))([w_mean, w_log_var])

# we instantiate these layers separately so as to reuse them later
decoder_h = Dense(intermediate_dim, activation='relu')
decoder_mean = Dense(original_dim, activation='tanh')
w_decoder = Dense(intermediate_dim, activation='relu')
w_decoder_mean = Dense(original_dim, activation='tanh')

def decode(z,w):
    h_decoded = decoder_h(z)
    x_decoded_mean = decoder_mean(h_decoded)
    w_decoded = w_decoder(w)
    w_decoded_mean = w_decoder_mean(w_decoded)
    x_decoded_mean = Lambda(lambda x: 0.5*(x+w_decoded_mean))(x_decoded_mean)
    return x_decoded_mean

x_decoded_mean = decode(z,w)




# Custom loss layer
class CustomVariationalLayer(Layer):
    def __init__(self, **kwargs):
        self.is_placeholder = True
        super(CustomVariationalLayer, self).__init__(**kwargs)

    def vae_loss(self, x, x_decoded_mean):
        xent_loss = original_dim * metrics.binary_crossentropy(x, x_decoded_mean)
        kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        kl_loss_w = - 0.5 * K.sum(1 + w_log_var - K.square(w_mean) - K.exp(w_log_var), axis=-1)
        #kl_loss_w = - 0.5 * K.sum(1 + w_log_var - K.square(w_mean) - K.exp(w_log_var), axis=-1)
        #kl_loss = tf.Print(kl_loss,
        #                   [K.get_variable_shape(xent_loss)],
        #                   'xent ssim ')
        return K.mean(xent_loss + kl_loss + kl_loss_w)

    def vae_frac_loss(self, x,x_decoded_mean):
        x = K.reshape(x,[-1,int(np.sqrt(original_dim)),int(np.sqrt(original_dim))])
        x_decoded_mean = K.reshape(x_decoded_mean,[-1,int(np.sqrt(original_dim)),int(np.sqrt(original_dim))])
        lvi_loss = original_dim * fbm_data.loss_logvarinc(x,x_decoded_mean,int(np.sqrt(original_dim)))
        kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        #kl_loss_w = - 0.5 * K.sum(1 + w_log_var - K.square(w_mean) - K.exp(w_log_var), axis=-1)
        #lvi_loss = tf.Print(lvi_loss,[lvi_loss],'lvi')
        return K.mean(lvi_loss + kl_loss )#+ kl_loss_w)

    def vae_ssim_loss(self, x,x_decoded_mean):
        #xent_loss = original_dim * metrics.binary_crossentropy(x, x_decoded_mean)
        dim1=int(np.sqrt(original_dim))
        x = K.reshape(x,[-1,dim1,dim1])
        x_decoded_mean = K.reshape(x_decoded_mean,[-1,dim1,dim1])

        ssim_loss = original_dim * fbm_data.loss_DSSIS_tf11(x,x_decoded_mean, False)#,int(np.sqrt(original_dim)))

        kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        kl_loss_w = - 0.5 * K.sum(1 + w_log_var - K.square(w_mean) - K.exp(w_log_var), axis=-1)
        #ssim_loss = tf.Print(ssim_loss,[K.get_variable_shape(xent_loss+kl_loss), K.get_variable_shape(kl_loss)],'xent ssim ')
        #kl_loss_w = - 0.5 * K.sum(1 + w_log_var - K.square(w_mean) - K.exp(w_log_var), axis=-1)
        #print xent_loss, ssim_loss, kl_loss
        return K.mean(ssim_loss + kl_loss + kl_loss_w)

    def call(self, inputs):
        x = inputs[0]
        x_decoded_mean = inputs[1]
        loss = self.vae_loss(x, x_decoded_mean)
        #loss = self.vae_frac_loss(x, x_decoded_mean)
        #loss = self.vae_ssim_loss(x, x_decoded_mean)
        self.add_loss(loss, inputs=inputs)
        # We won't actually use the output.
        return x

y = CustomVariationalLayer()([x, x_decoded_mean])
vae = Model(x, y)
vae.compile(optimizer='rmsprop', loss=None)




x_train = x_train0.astype('float32') / 255.
x_test = x_test0.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
y_train = y_train0
y_test = y_test0

vae.fit(x_train,
        shuffle=True,
        epochs=epochs,
        verbose=2,
        batch_size=batch_size,
        validation_data=(x_test, x_test))

# build a model to project inputs on the latent space
encoder = Model(x, z_mean)

# display a 2D plot of the digit classes in the latent space
x_test_encoded = encoder.predict(x_test, batch_size=batch_size)
plt.figure(figsize=(6, 6))
plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=y_test)
plt.colorbar()
plt.show()

# build a digit generator that can sample from the learned distribution
decoder_input = Input(shape=(latent_dim,))
decoder_input_w = Input(shape=(latent_dim_w,))
#_h_decoded = decoder_h(decoder_input)
#_x_decoded_mean = decoder_mean(_h_decoded)
_x_decoded_mean = decode(decoder_input,decoder_input_w)
generator = Model([decoder_input, decoder_input_w], _x_decoded_mean)

# display a 2D manifold of the digits
n = 15  # figure with 15x15 digits
digit_size = int(np.sqrt(original_dim))
figure = np.zeros((digit_size * n, digit_size * n))
h_figure = np.zeros((n,n,2))
# linearly spaced coordinates on the unit square were transformed through the inverse CDF (ppf) of the Gaussian
# to produce values of the latent variables z, since the prior of the latent space is Gaussian
grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
grid_y = norm.ppf(np.linspace(0.05, 0.95, n))
##
for i, yi in enumerate(grid_x):
    for j, xi in enumerate(grid_y):
        z_sample = np.array([[xi, yi]])
        w_sample = np.random.randn(1,latent_dim_w)*1
        # randomize w instead of z as the manifold parameters
        # here we should see a disoriented behaviour
        #z_sample = np.random.randn(1,latent_dim)*1
        #w_sample = np.concatenate([np.array([[xi, yi]]), [np.random.randn(latent_dim_w-2)*1]],axis=1)

        x_decoded = generator.predict([z_sample,w_sample])
        digit = x_decoded[0].reshape(digit_size, digit_size)
        h_figure[i,j,:] = fbm_data.hurst2d(digit,max_tau=5)
        figure[i * digit_size: (i + 1) * digit_size,
               j * digit_size: (j + 1) * digit_size] = digit

#plt.figure(figsize=(10, 10))
figure=figure-np.min(figure)
figure=figure/np.max(figure)
_,pp=plt.subplots(2,2)
pp[0,0].imshow(figure, cmap='gray',interpolation='none')
#.show()
res=pp[0,1].imshow(h_figure[:,:,0],cmap='gray',interpolation='none')
plt.colorbar(res,ax=pp[0,1])
pp[0,1].set_title('H')
res=pp[1,0].imshow(h_figure[:,:,1],cmap='gray',interpolation='none')
plt.colorbar(res,ax=pp[1,0])
pp[1,0].set_title('R^2')

#plt.show()
