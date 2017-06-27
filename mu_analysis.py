import cPickle as pickle
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model

from sklearn.decomposition import PCA



all_pred, all_enc, y = pickle.load(open('all_pred.bin','r'))
y = pickle.load(open('yy.bin','r'))
y = np.array(y)
train_vars, train_vars_std = pickle.load(open('train_vars.bin','r'))
##


fig = plt.figure(0)
plt.hold(False)
ax = fig.add_subplot(111, projection='3d')
features = {}
def standardize(x):
    x=np.array(x)
    x[np.isnan(x)]=0.0
    x=x-np.min(x)
    x=x/np.max(x)
    return x
features['H'] =       standardize([ np.float32(u) for  u in y[:,0] ]) # H
features['Kurt'] =    standardize([ np.log(1+np.float32(u)) for  u in y[:,1] ]) # log kurt
features['StdCoh'] =  standardize([ np.log(1+np.float32(u)) for  u in y[:,3] ]) # std log coh
features['MeanCoh'] = standardize([ np.log(1+np.float32(u)) for  u in y[:,4] ]) # mean log coh

chosen = features['H']

color = [plt.cm.jet(c) for c in chosen]
map = {}; k=1
classes = ['wool','cotton','cracker']
subclasses = ['a','b','c','d']
markers0 = ['+','x','.','^']
colors = ['r','g','b','c','m','y','k']
markers = ['o' for i in range(len(color))]
markers={}
for ii,i in enumerate(classes):
    kk=0
    for jj,j in enumerate(subclasses):
        map[i+j]=k
        #markers[i+j]=markers0[jj] # for markers of subclass
        markers[i+j]=markers0[ii] # for markers of class
    k+=1
#color = [colors[map[c]] for c in y[:,2]] # color by class
def get_linreg(x,y):
    clf = linear_model.LinearRegression()
    clf.fit(x,y)
    return clf

marker = [markers[c] for c in y[:,2]]
use_ae = False
if use_ae:
    PCA_ = PCA(n_components=5)
    pca = PCA_.fit(all_enc)
    transformed = PCA_.transform(all_enc)

    x1=transformed[:,0]
    x2=transformed[:,1]
    x3=transformed[:,2]
    X = np.array([ [a1,a2] for a1,a2 in zip(x1,x2) ])
    Y = x3
    plt.clf()
    ax = fig.add_subplot(111, projection='3d')
    plt.hold(True)
    #color = [1.0/max(color) for c in color]
    for xx1,xx2,xx3,c,m in zip(x1,x2,x3,color,marker):
        ax.scatter(xx1,xx2,zs=xx3,s=30,c=c,marker=m)
    #plt.scatter(x1,x2,zs=x3,s=20,c=color)

else:
    train_vars_compressed = []
    for k, cur in enumerate(train_vars):
    #for k, cur in enumerate(train_vars_std):
        #if k!=1:
        #    continue
    #cur = train_vars[5]
        PCA_ = PCA(n_components=10)
        print len(cur)
        pca = PCA_.fit(cur)
        try:
            low_ind = np.where(PCA_.explained_variance_ratio_<0.01)[0][0]
        except:
            low_ind = -1
        print 'scale',k,'low ind',low_ind
        transformed = PCA_.transform(cur)
        if low_ind>-1:
            transformed[:,low_ind:]=0
        inverse = PCA_.inverse_transform(transformed)
        train_vars_compressed.append(inverse)

        x1=transformed[:,0]
        x2=transformed[:,1]
        x3=transformed[:,2]
        # linreg for color (i.e. feature w.r.t x vector
        coors = {}
        for kk,feat in features.iteritems():
            res = get_linreg(np.stack([x1,x2,x3]).T,feat)
            print 'R^2', res.score(np.stack([x1,x2,x3]).T,feat)
            coors[kk] = res.coef_/np.linalg.norm(res.coef_)*np.max(np.stack([x1,x2,x3]))/2
        plt.clf()
        plt.plot(PCA_.explained_variance_ratio_)
        plt.pause(0.1)
        plt.clf()
        fig = plt.figure(0)
        plt.hold(True)
        ax = fig.add_subplot(111, projection='3d')
        #plt.hold(True)
        for xx1,xx2,xx3,c,m in zip(x1,x2,x3,color,marker):
            #pass
            ax.scatter(xx1,xx2,zs=xx3,s=30,c=c,marker=m,facecolor=c)
        #
        coorsmat = np.array(coors.values())
        print 'coors rank', np.linalg.matrix_rank(coorsmat)
        linestyles=['-','--',':','-.']
        for (val, kk), ls in zip(coors.iteritems(), linestyles):
            plt.plot([0, kk[0]],[0, kk[1]],[0, kk[2]],ls=ls,lw=3)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(str(k))
        plt.pause(0.1)
        plt.show()

##
pickle.dump([train_vars_compressed, train_vars_std],open('train_vars_comp.bin','w'))
