import matplotlib.pyplot as plt
import numpy as np
from numpy import array
import cPickle as pickle
'''
# load 'graph_pos.bin'
see which values are > 0.07~ in the vertical axis
see their classification in the euc v int cases 
etc.
'''
def graph_pos_analysis():
    x=pickle.load(open('graph_pos.bin','r'))
    xx=np.array(x.values())

    plt.plot(xx[:,0],xx[:,1],'.')


    t=[]
    t.append(xx[:,1]>0.03)
    t.append(xx[:,1]<0.05)
    t.append(xx[:,0]<-0.1)
    t=np.array(t).T
    print t
    sel=np.zeros_like(xx[:,1])
    for i in range(len(sel)):
        sel[i]=t[i,:].all()
    sel=[True if s==1 else False for s in sel]
    print sel
    plt.plot(xx[sel,0],xx[sel,1],'.')

    print np.array(x.keys())[sel]
    plt.show()

def cluster_mat_analysis(cluster_mat=None):
    if cluster_mat is None:
        cluster_mat = {'int': array([[64, 17, 14, 39,  0],
           [ 0, 76,  0, 53,  0],
           [ 2,  0, 36,  0,  0],
           [19,  0, 38,  2,  0],
           [ 0,  0,  0,  0,  1]]),
            'euc': array([[55,  2,  3, 38,  0],
                          [28,  0, 43,  2,  0],
                          [ 1, 84,  0, 47,  0],
                          [ 0,  7,  0,  7,  1],
                          [ 1,  0, 42,  0,  0]])}

    ent_mat = dict(zip(cluster_mat.keys(),[0,0]))
    for i,v in cluster_mat.iteritems():
        v0 = np.array([ 1.0*x/np.sum(x) for x in v ])
        v1 = np.array([ -x*np.log2(x) for x in v0 ])
        v1[np.isnan(v1)]=0
        ent_mat[i] = np.mean(v1)
    print 'ent_mat',ent_mat

    print 'int'
    print cluster_mat['int']
    print 'euc'
    print cluster_mat['euc']

if __name__=='__main__':
    cluster_mat_analysis()
