from fbm.fbm import fbm
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
import pandas
from keras import backend as K
from keras.models import Model
from keras.layers import Dense
from keras.layers import Lambda
from keras.layers import Flatten
from keras.layers import Conv1D, MaxPool1D, UpSampling1D
from keras.layers import Input
import tensorflow as tf
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def calcNorms(V):
    r=K.square(V)
    r=K.sum(r,1)
    r = K.log(r)
    return r
#def baseline_model(n=256):
##
n=512
activation='tanh'
inputs = Input(shape=(n,1))
"""
x = Conv1D(filters=10,kernel_size=10,padding='same',input_shape=(256,1))(inputs)
x = Lambda(calcNorms)(x)
h = Dense(5,input_dim=10, kernel_initializer='normal')(x)
y = Dense(10, kernel_initializer='normal', activation=activation)(h)
y = Dense(20, kernel_initializer='normal', activation=activation)(y)
y = Dense(50, kernel_initializer='normal', activation=activation)(y)
y = Dense(100, kernel_initializer='normal', activation=activation)(y)
print K.int_shape(y)
out = Dense(n, kernel_initializer='normal', activation=activation, name='out')(y)
"""
padding = 'same'
x = Conv1D(filters=4,kernel_size=3,padding=padding,input_shape=(256,1),activation=activation)(inputs)
x = MaxPool1D(2,padding=padding)(x)
x = Conv1D(filters=4,kernel_size=3,padding=padding,activation=activation)(x)
x = MaxPool1D(2,padding=padding)(x)
x = Conv1D(filters=4,kernel_size=3,padding=padding,activation=activation)(x)
x = MaxPool1D(2,padding=padding)(x)
x = Conv1D(filters=4,kernel_size=3,padding=padding,activation=activation)(x)
x = MaxPool1D(2,padding=padding)(x)
x = Conv1D(filters=2,kernel_size=3,padding=padding,activation=activation)(x)
x = MaxPool1D(2,padding=padding)(x)
x = Conv1D(filters=2,kernel_size=3,padding=padding,activation=activation)(x)
x = MaxPool1D(2,padding=padding)(x)
x = Conv1D(filters=2,kernel_size=3,padding=padding,activation=activation)(x)
encoded = MaxPool1D(2,padding=padding)(x)

x = Conv1D(filters=2,kernel_size=3,padding=padding,activation=activation)(encoded)
x = UpSampling1D(2)(x)
x = Conv1D(filters=2,kernel_size=3,padding=padding,activation=activation)(x)
x = UpSampling1D(2)(x)
x = Conv1D(filters=2,kernel_size=3,padding=padding,activation=activation)(x)
x = UpSampling1D(2)(x)
x = Conv1D(filters=4,kernel_size=3,padding=padding,activation=activation)(x)
x = UpSampling1D(2)(x)
x = Conv1D(filters=4,kernel_size=3,padding=padding,activation=activation)(x)
x = UpSampling1D(2)(x)
x = Conv1D(filters=4,kernel_size=3,padding=padding,activation=activation)(x)
x = UpSampling1D(2)(x)
x = Conv1D(filters=4,kernel_size=3,padding=padding,activation=activation)(x)
x = UpSampling1D(2)(x)
out = Conv1D(filters=1,kernel_size=3,padding=padding,activation=activation)(x)
out = Lambda(lambda x : K.squeeze(x,2) )(out)
print K.int_shape(out)
gradfilt = np.array([1,-1],dtype=np.float32)
gradfilt = np.expand_dims(gradfilt,1)
gradfilt = np.expand_dims(gradfilt,2)
def normgrad(x):
    print 'normgrad'
    print gradfilt.shape
    res = tf.nn.conv1d(K.expand_dims(x,2),gradfilt,stride=1,padding='VALID')
    #res = K.squeeze(res,2)
    print K.int_shape(res)
    res = K.sum(K.square(res),1)
    print K.int_shape(res)
    return res
out_smoothness = Lambda(normgrad)(out)
print K.int_shape(out_smoothness)
#out = Lambda(sq)(out)
# x is now 1-dimensional


##
def shuffle_weights(model, weights=None):
    """Randomly permute the weights in `model`, or the given `weights`.
    This is a fast approximation of re-initializing the weights of a model.
    Assumes weights are distributed independently of the dimensions of the weight tensors
      (i.e., the weights have the same distribution along each dimension).
    :param Model model: Modify the weights of the given model.
    :param list(ndarray) weights: The model's weights will be replaced by a random permutation of these weights.
      If `None`, permute the model's current weights.
    """
    np.random.seed(0)
    if weights is None:
        weights = model.get_weights()
    weights = [np.random.rand(len(w.flat)).reshape(w.shape) for w in weights]
    # Faster, but less random: only permutes along the first dimension
    # weights = [np.random.permutation(w) for w in weights]
    model.set_weights(weights)

## ################################################

np.random.seed(0)
N=6000
X = []
H = np.linspace(0.2,0.8,N)# [0.5]*N
for i in H:
    fbmr,fgnr,times= fbm(n-1,i,L=1)
    #fbmr = fbmr-np.min(fbmr)
    #fbmr = fbmr/np.max(fbmr)
    X.append(fbmr)
X0=np.array(X)
H = np.array(H)
Y = H
X=np.expand_dims(X0,2) # to make input of size (n,1) instead of just (n)
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X0,Y,test_size=0.2,random_state=0)

## ################################################
np.random.seed(0)
smoothness = np.array([0.00001]*len(H))
Y= [X0]
print 'training...'
model = Model(inputs=inputs,outputs=[out])
#shuffle_weights(model) ### this does give replicated results but uses bad initialization
#compile
model.compile(loss='mean_squared_error',optimizer='adam')

model.fit(X, Y ,batch_size=100,epochs=200)
train_loss = model.evaluate(X, Y, batch_size=100)
#test_loss = model.evaluate(X_test, y_test, batch_size=100)
#plt.plot(times,fbmr)
#plt.show()
print
print 'train loss',train_loss#,'test loss',test_loss

#kfold = KFold(n_splits=10, random_state=seed)
#results = cross_val_score(estimator, X, H, cv=kfold)
## print stuff ####################################

for l in model.layers:
    w = l.get_weights()
    print 'layer',l
    for k in w:
        if len(k.shape)==3: # conv
            plt.plot(np.squeeze(k).T)
            plt.show(block=True)

        #print k.shape
##
w = model.layers[3].get_weights()
print w
plt.plot(np.squeeze(w[1]))
    #raw_input()

## try to predict series
i=800
true_h = H[i]
pred = model.predict(np.expand_dims(X[i],0))
plt.hold(False)
plt.plot(pred[0],'r')
plt.hold(True)
plt.plot(X[i],'b')
#plt.title('true h:%s, est h:%s'%(true_h,pred_h[0][0]))
plt.title('true h:%s'%(true_h))
plt.show()







