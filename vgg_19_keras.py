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
import cv2, numpy as np

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
        g = np.dot(act_mat.transpose(),act_mat)/imsz[i]**2
        G.append(g)
    return G

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
                print 'loading layer',layer.name
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
            print 'popping', model.layers[-1]
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


def synTexture(im):

    use_caffe= True# True
    if not use_caffe:
        im[:,:,0] -= 103.939
        im[:,:,1] -= 116.779
        im[:,:,2] -= 123.68
    else: # caffe
        im = im/255.0
        mm=np.array([ 0.40760392,  0.45795686,  0.48501961])
        im[:,:,0]=im[:,:,0]-np.mean(im[:,:,0])+mm[0] # b
        im[:,:,1]=im[:,:,1]-np.mean(im[:,:,1])+mm[1] # g
        im[:,:,2]=im[:,:,2]-np.mean(im[:,:,2])+mm[2] # r


    #im = im.transpose((2,0,1))
    im = np.expand_dims(im, axis=0)
    # Test pretrained model
    model = VGG_19_1('vgg19_weights_tf_dim_ordering_tf_kernels.h5',onlyconv=True,caffe=use_caffe)
    #model = VGG_19_1('dataout.h5',onlyconv=True)
    #vgg19_weights.h5
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    print 'compiling'
    #model.compile(optimizer=sgd, loss='categorical_crossentropy')
    print 'predicting'
    out = model.predict(im)

    #conv_layers = [1,3,6,8,11,13,15,17,20,22,24,26,29,31,33,35]
    conv_layers = [0,1,3,4,6,7,8,9,11,12,13,14,16,17,18,19]
    activations = get_activations(model,im,conv_layers)
    G0 = get_gram_matrices(activations)
    G0_symb, N_l, M_l = get_gram_matrices_symb(model,sel=conv_layers)

    errf = get_gram_error(G0,G0_symb,N_l,M_l)
    #print errf
    grads = K.gradients(errf,model.input)
    #opt = Adam()

    #opt.get_gradients
    #updates = opt.get_updates([model.input],[],[errf])
    #train = K.function([model.input],[errf, model.input],updates=updates)
    coef=0.5 if use_caffe else 128.0
    im_iter = np.random.randn(im0.shape[0],im0.shape[1],im0.shape[2])*coef
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
    maxiter = 2000
    method = 'L-BFGS-B'
    #method = 'BFGS'
    cur_iter = 1
    def callback(x):
        global cur_iter
        plt.imshow(np.reshape(x,imsize)[::-1])
        #print 'random number',np.random.rand(1)
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
    res = minimize(min_fun,im_iter.flatten().astype(np.float64),
                   jac=grad_fun,method=method,bounds=bounds,callback=callback,
                   options={'disp': True, 'maxiter':maxiter,
                            'maxcor': m, 'ftol': 0, 'gtol': 0})

    pp[0,0].imshow(im_iter)
    pp[0,1].imshow(res[::-1])
    plt.show(block=False)
    plt.pause(0.01)
    raw_input('done')
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



if __name__ == "__main__":
    kth_path = '/home/ido/combined_model_classification/KTH-TIPS2-b/wool/sample_a/'
    images=['pebbles.jpg','cat.jpg', '22a-scale_3_im_3_col.png']
    sel_img = images[2]


    im = cv2.resize(cv2.imread(sel_img), (224, 224)).astype(np.float32)
    im0=im






    #print np.argmax(out)
