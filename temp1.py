from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
import pandas
from keras import backend as K
from keras.models import Model
from keras.layers import Dense
from keras.layers import Lambda
from keras.layers import Flatten
from keras.layers import Conv1D
from keras.layers import Input
import tensorflow as tf
x_batch = K.ones(shape=(32, 256, 10))
y_batch = K.ones(shape=(32, 256, 10))

r=K.square(x_batch)
r=K.sum(r,1)
print K.int_shape(r)
#xy_batch_dot = K.batch_dot(x_batch, K.permute_dimensions(y_batch,[0,2,1]), axes=[1,2])
#xy_batch_dot = tf.diag_part(xy_batch_dot)
print K.int_shape(xy_batch_dot)
