import matplotlib.pyplot as plt
import cPickle as pickle
import numpy as np
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.ticker as ticker
from draw_ellipse import DrawEllipse
from sklearn.manifold import SpectralEmbedding
from scipy.misc import imresize
class LearnMetric():
    def __init__(self,emb_dim=10,intrinsic_dim=5,sqeps=0.1,genericEmbed=None):
        self.emb_dim=emb_dim
        self.intrinsic_dim=intrinsic_dim
        self.sqeps=sqeps
        if genericEmbed is not None:
            self.genericEmbed = genericEmbed
        else:
            print 'using PCA as genericEmbed'
            self.genericEmbed = self.pcaEmbed
            #print 'using Eigenmaps as genericEmbed'
            #self.genericEmbed = self.eigEmbed
        self.lam=1

    def pcaEmbed(self,data, s, d, sqeps):
        PCA_ = PCA(n_components=s)
        PCA_.fit(data)
        pca_coeffs = PCA_.transform(data)

        return pca_coeffs

    def eigEmbed(self,data, s, d, sqeps):
        lapeig = SpectralEmbedding(n_components=s,affinity='rbf',gamma=100)
        #lapeig = SpectralEmbedding(n_components=s,affinity='nearest_neighbors',n_neighbors=10)
        return lapeig.fit_transform(data)

    def getNeighGraph(self,data):
        n = data.shape[0]
        m = data.shape[1]
        W = np.zeros((n,n))
        fun = lambda x,y: np.exp(-1.0/self.sqeps**2 * np.sum((x-y)**2) )
        for i in range(n):
            for j in range(n):
                W[i,j] = fun(data[i,:],data[j,:])

        return W

    def getLaplacian(self,n,W):
        # graph laplacian
        c = 0.25
        onev = np.ones(W.shape[0])
        Dl = np.diag(W.dot(onev))**(1.0/self.lam)
        Wt = Dl.dot(W).dot(Dl)
        Dt = np.diag(Wt.dot(onev))
        L = (c*self.sqeps)**-1 * ( np.linalg.inv(Dt).dot(Wt) - np.eye(n) )

        return L

    def pseudoInv(self,mat):
        # mat should be symmetric for this to be valid.

        d = self.intrinsic_dim
        vals,vects = np.linalg.eig(mat)
        order = np.argsort(vals)[::-1]
        vals = vals[order]
        vects = vects[:,order]

        vals = np.diag(1.0/vals[:d])
        vects = vects[:,:d]
        return np.real(vects.dot(vals).dot(vects.T))


    def learnMetric(self,data):
        n = data.shape[0]

        W = self.getNeighGraph(data)

        Lap = self.getLaplacian(n,W)

        genEmb = self.genericEmbed(data,self.emb_dim,self.intrinsic_dim,self.sqeps)
        # should be a matrix where each row is an entry

        hh = np.zeros((self.emb_dim,self.emb_dim,n))
        H = np.zeros((self.emb_dim,self.emb_dim,n))
        for l in range(n):
            for i in range(self.emb_dim):
                for j in range(self.emb_dim):
                    fi = genEmb[:,i]
                    fj = genEmb[:,j]
                    xx = fi*Lap.dot(fj)

                    hh[i,j,:] = 0.5* ( Lap.dot(fi*fj) -fi*Lap.dot(fj) -fj*Lap.dot(fi) )

        for l in range(n):
            h = np.squeeze(hh[:,:,l])
            #print h.shape
            H[:,:,l] = self.pseudoInv(h)
        return genEmb,H


if __name__=='__main__':
    dataset = 'kth'
    load_file = 'train_vars_'+dataset+'.bin'
    used_data_fname = 'used_data_'+dataset+'.bin'
    x_train0, x_test0, y_train0, y_test0 = pickle.load(open(used_data_fname,'r'))

    train_vars_compressed, train_vars_std_compressed, train_vars_s_compressed = pickle.load(open(load_file,'r'))
    use_one_response = False
    if use_one_response:
        data = train_vars_compressed[0]
    else:
        data = np.concatenate(train_vars_compressed,axis=1)

    print 'data shape',data.shape
    Metric = LearnMetric(emb_dim=3,intrinsic_dim=3,sqeps=5)

    genEmb,H = Metric.learnMetric(data)
    print 'embedding shape', H.shape
    x1=genEmb[:,0]
    x2=genEmb[:,1]
    x3=genEmb[:,2]
    """
    fig=plt.figure(0)
    for k1 in range(2):
        for k2 in range(2):
            #fig = plt.subplot(2,2,k1*2+k2)
            ax = fig.add_subplot('22%d'%(k1*2+k2+1), projection='3d')
            ax.set_title('22%d'%(k1*2+k2+1))
            #plt.hold(True)
            #ax.scatter(x1,x2,zs=x3)
            plt.hold(True)
            h1 = H[k1,k2,:]
            #h1 = np.log(H[0,0,:])
            h1 = h1 - np.min(h1)
            h1 = h1 / np.max(h1)
            h1 = plt.cm.jet(h1)
            for xx1,xx2,xx3,h in zip(x1,x2,x3,h1):
                plt.scatter(xx1,xx2,zs=xx3,c=h)
    """
    plt.rcParams['text.usetex'] = True
    fig = plt.figure(0)

    ax = fig.add_subplot('111',projection='3d')
    #ax.set_title('ellipses')
    plt.hold(True)

    radi = []
    for i, xx1,xx2,xx3 in zip(range(H.shape[2]), x1, x2, x3):
        A = H[:,:,i]
        _, s, _ = np.linalg.svd(A)
        radi.append(1.0/np.sqrt(s))
        #lam,vec = np.linalg.eig(h)
    radi = np.array(radi)
    radi = np.std(radi,axis=1)
    radi = radi - np.min(radi)
    radi = radi / np.max(radi)
    radi = plt.cm.jet(radi)
    for r, xx1,xx2,xx3,i  in zip(radi, x1, x2, x3, range(H.shape[2])):
        #plt.scatter(xx1,xx2,zs=xx3,c=r)
        A = H[:,:,i]
        center = [xx1,xx2,xx3]
        de = DrawEllipse(A,center,scale=0.1,color=r)
        de.plot(ax)

        if not (i%30):
            cur = x_train0[i]
            #ax0 = fig.add_axes([xx1,xx2,xx1+1,xx2+1])
            rr = 0.01
            stepX, stepY = 2*2*rr / cur.shape[0], 2*rr / cur.shape[1]
            X1 = np.arange(-2*rr, 2*rr, stepX)
            Y1 = np.arange(-rr, rr, stepY)
            X1, Y1 = np.meshgrid(xx1+X1, xx3+Y1)
            #print np.min(cur),np.max(cur)
            ax.plot_surface(X1,xx2,Y1,rstride=2,cstride=2,facecolors=plt.cm.gray(cur/255.0),alpha=1.0)
                #imshow(imresize(cur,(5,5)), cmap=plt.cm.BrBG, interpolation='nearest', origin='lower', extent=[0,1,0,1])
            #fig.figimage(imresize(cur,(5,5)),xo=xx1,yo=xx2,cmap=plt.cm.gray)
            #ax0.imshow(imresize(cur,(0.1,0.1)))

    plt.xlabel('PC1')
    plt.ylabel('PC2')
    ax.set_zlabel('PC3')

    tick = ticker.ScalarFormatter(useMathText=True)
    #tick.FormatStrFormatter('%1.0e')
    #ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%1.0e'))
    #ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%1.0e'))
    #ax.zaxis.set_major_formatter(ticker.FormatStrFormatter('%1.0e'))
    #ax.xaxis.set_major_formatter(tick)
    #ax.yaxis.set_major_formatter(tick)
    #ax.zaxis.set_major_formatter(tick)
    ax.ticklabel_format(style='sci')

    save = False
    if save:
        plt.savefig('dataset_%s_metric1.pdf'%dataset)


    # see also Laplacian eigenmaps, for comparison
    lapeig = SpectralEmbedding(n_components=3,affinity='rbf',gamma=20)
    lapeig_res = lapeig.fit_transform(data)
    ax2 = plt.figure(1).add_subplot(111,projection='3d')
    ax2.scatter(lapeig_res[:,0],lapeig_res[:,1],zs=lapeig_res[:,2])



    plt.show()




