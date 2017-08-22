import matplotlib
#matplotlib.use('GTKAgg')#
#matplotlib.use("Qt4Agg") # Not working
import matplotlib.pyplot as plt
import cPickle as pickle
import numpy as np
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.ticker as ticker
#from draw_ellipse import DrawEllipse # to use with matplotlib
from sklearn.manifold import SpectralEmbedding
from scipy.misc import imresize, imsave
from mayavi import mlab
from plot_3d_mayavi import DrawEllipse, ImageOverlay
import scipy.io as sio
import networkx as nx
from os import mkdir
from shutil import rmtree
from analyze_metric_res import cluster_mat_analysis
class LearnMetric:
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
        PCA_full_ = PCA(n_components=20)
        PCA_full_.fit(data)

        PCA_ = PCA(n_components=s)
        PCA_.fit(data)
        pca_coeffs = PCA_.transform(data)
        #print PCA_.explained_variance_
        #print PCA_.explained_variance_ratio_
        return pca_coeffs, PCA_full_.explained_variance_ratio_

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
        print 'mean weight',np.mean(W),'weight std',np.std(W)
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

        genEmb, explained_var = self.genericEmbed(data,self.emb_dim,self.intrinsic_dim,self.sqeps)
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
        return genEmb,H, explained_var

    def calcDistance(self,data,genEmb,explained_var,M):#,p_from=2,p_to=10,
                     #imgs=None, features=None, feature_names=None):
        n = M.shape[2]
        dists_deg = np.zeros((n,n))
        dists_deg = dists_deg*np.nan
        dists_int = dists_deg*np.nan
        one_dist = lambda f1,f2,h: np.dot(f1-f2,h).dot(f1-f2)
        G = nx.Graph()
        G.add_nodes_from(range(n))
        degenerated = np.diag(np.ones(M.shape[0]))
        for i in range(n):
            for j in range(i+1,n):
                q1, q2 = data[i,:], data[j,:]
                p1, p2 = genEmb[i,:], genEmb[j,:]
                h1, h2 = M[:,:,i], M[:,:,j]
                #if not np.isnan(h1).any():
                #    print np.linalg.eig(h1)

                dists_int[i,j] = 0.5*( np.sqrt(one_dist(p1,p2,h1)) + np.sqrt(one_dist(p1,p2,h2)) )
                # scale by explained_var
                dists_deg[i,j] = np.sqrt( np.sum( explained_var * (p1-p2)**2 ) )
        max_nei = 3
        dists_good = np.zeros((n,n))
        dists_int_good = np.zeros((n,n))
        for i in range(n):
            bad = np.argsort(dists_deg[i,:])[max_nei:]
            good = np.argsort(dists_deg[i,:])[:max_nei]
            dists_good[i,[bad]]=False
            dists_good[i,[good]]=True

            bad = np.argsort(dists_int[i,:])[max_nei:]
            good = np.argsort(dists_int[i,:])[:max_nei]
            dists_int_good[i,[bad]]=False
            dists_int_good[i,[good]]=True

        for i in range(n):
            for j in range(i+1,n):
                # calc intrinsic distance between data(i) and data(j)
                p1, p2 = genEmb[i,:], genEmb[j,:]
                h1, h2 = M[:,:,i], M[:,:,j]
                q1, q2 = data[i,:], data[j,:]
                dists= 0.5*( np.sqrt(one_dist(p1,p2,h1)) + np.sqrt(one_dist(p1,p2,h2)) )
                dists_deg = np.sqrt( np.sum( (p1-p2)**2 ) )
                #print dists, dists_deg
                if dists_int_good[i,j]:
                    G.add_edge(i,j,{'int_dist':dists,'euc_dist':dists_deg})

        weights = ['int_dist','euc_dist']
        return G, weights

    def kmeans_assign(self,G,weight,centroid_nodes,n,genEmb,disp_i=None):

        def getAssignment(G,w,centroid_nodes):
            node_center_distance = {}
            for centroid in centroid_nodes:
                node_center_distance[centroid] = nx.shortest_path_length(G,source=centroid,weight=w)

            assignments = {}
            for centroid in centroid_nodes:
                assignments[centroid]=[]

            for node in G.nodes():
                minimal = {'val':np.inf,'ind':-1}
                for centroid in centroid_nodes:
                    if node_center_distance[centroid][node]<minimal['val']:
                        minimal['val']=node_center_distance[centroid][node]
                        minimal['ind']=centroid
                assignments[minimal['ind']].append(node)

            return assignments

        assignments = getAssignment(G,weight,centroid_nodes)

        if disp_i is not None:
            # display
            pos = dict(zip(range(n),genEmb[:,:2])) # 2D for display

            pickle.dump(pos,open('graph_pos.bin','w'))
            print 'saved pos to file graph_pos.bin'

            colors = ['r','g','b','y','m']
            if disp_i <> True:
                plt.figure(disp_i+1)
            for centroid,color in zip(centroid_nodes,colors):
                nx.draw(G,pos,nodelist=assignments[centroid],node_color=color,width=0.01,node_size=2)
            nodes = nx.draw_networkx_nodes(G,pos,nodelist=centroid_nodes,edgelist=[],
                            node_color=colors[:len(centroid_nodes)],
                            node_size=15,node_shape='^')
            nodes.set_edgecolor('k')
            nodes.set_linewidth(0.3)
            #nx.draw(G,pos,)

            #plt.show()
        return assignments

    def kmeans_update(self,G,weight,assignments,n,genEmb,disp_i=None):
        def getCentroids(G,w,assignments):
            center_trans = dict(zip(assignments.keys(),assignments.keys()))
            for center, assigned in assignments.iteritems():
                best_dist = np.inf
                best_src = 0
                for src in assigned:
                    dists=[]
                    dists = nx.shortest_path_length(G,source=src,weight=w)
                    # intersect dists with assigned
                    dists = [ v for i,v in dists.iteritems() if i in assigned ]

                    if best_dist > np.sum(dists):
                        best_dist = np.sum(dists)
                        best_src = src
                center_trans[center] = best_src

            return center_trans.values()

        centroids = getCentroids(G,weight,assignments)

        if disp_i is not None:
            # display
            pos = dict(zip(range(n),genEmb[:,:2])) # 2D for display
            colors = ['r','g','b','k','y']
            plt.figure(disp_i+1)
            nx.draw(G,pos,nodelist=centroids,edgelist=[],
                    node_color=colors[:len(centroids)],
                    node_size=30,node_shape='^')
            nx.draw(G,pos,node_color='k',width=0.01,node_size=2)

        return centroids


    def exampleDistance(self,data,genEmb,explained_var,M,p_from=2,p_to=10,
                        imgs=None, features=None, feature_names=None):

        G, weights = self.calcDistance(data,genEmb,explained_var,M)
        n = M.shape[2]
        # see path with images
        #pos = nx.spring_layout(G,weight='euc_dist')
        #p_to = 100
        #p_from = 10
        p_to = 20

        pp_sets = []
        pp_sets.append([2,20])
        pp_sets.append([5,30])
        pp_sets.append([10,30])
        pp_sets.append([10,60])
        pp_sets.append([30,80])
        for p_from,p_to in pp_sets:
            print 'CURRENT: p_from,p_to=',p_from,p_to
            save_fname = lambda x,ext: 'res_geometry/%d_%d_%s.%s'%(p_from,p_to,x,ext)
            plt.figure(2).clf()
            plt.figure(3).clf()
            plt.figure(4).clf()
            plt.figure(1)
            plt.clf()

            pos = dict(zip(range(n),genEmb[:,:2])) # 2D for display
            nx.draw(G,pos,node_color='k',width=0.3,node_size=2)
            # draw path in red

            colors = ['r','b']
            styles = ['solid','dashed']
            use_imgs = imgs is not None
            if use_imgs:
                path_imgs = []
                path_feats = []
                for i in colors:
                    path_imgs.append([])
                    path_feats.append([])



            paths = []
            for w,c,s in zip(weights,colors,styles):
                path = nx.shortest_path(G,source=p_from,target=p_to,weight=w)
                paths.append(path)
                path_edges = zip(path,path[1:])
                nx.draw_networkx_nodes(G,pos,nodelist=path,node_color=c,width=1.2,node_size=8,style=s)
                nx.draw_networkx_edges(G,pos,edgelist=path_edges,edge_color=c,width=1.2,node_size=8,style=s)

            plt.axis('equal')
            #plt.title('int dist=r, euc_dist=b')

            plt.savefig(save_fname('graph','pdf'))
            plt.rcParams['axes.labelsize'] = 'xx-large'
            plt.rcParams['ytick.labelsize'] = 'xx-large'
            plt.rcParams['xtick.labelsize'] = 'xx-large'
            plt.rcParams['lines.linewidth'] = 4
            plt.rcParams['lines.markersize'] = 15
            if use_imgs:
                tot_feats = 3
                prev_stacked = None
                for i, (path, w) in enumerate(zip(paths,weights)):
                    path_im = imgs[path]
                    feats = features[path]
                    emb = genEmb[path,:]
                    #print emb
                    stacked_im = np.hstack(path_im)
                    stacked_im[:,::path_im[0].shape[1]]=255.0
                    if prev_stacked is None:
                        prev_stacked = stacked_im
                    elif prev_stacked.shape[1]-stacked_im.shape[1]>0:
                        stacked_im = np.pad(stacked_im,( (0,0), (0,prev_stacked.shape[1]-stacked_im.shape[1])),
                                            mode='constant',constant_values=(0.,0.))
                    imsave(save_fname('imgs_%s'%w,'png'),stacked_im)
                    feats = np.array(feats)
                    plt.figure(2)
                    plt.subplot(len(weights),1,i+1)
                    plt.imshow(stacked_im,cmap=plt.cm.gray)
                    plt.title(w[:3])
                    plt.figure(3)

                for sel,feat in enumerate(feature_names[:tot_feats]):
                    plt.subplot(tot_feats,1,sel+1)
                    feats_inc = features[paths[0]]
                    feats_euc = features[paths[1]]
                    plt.plot(feats_inc[:,sel],'r.-')
                    plt.plot(feats_euc[:,sel],'b.--')
                    plt.title(feat)

                    # save
                    plt.figure(10)
                    plt.plot(feats_inc[:,sel],'r.-')
                    plt.plot(feats_euc[:,sel],'b.--')
                    plt.savefig(save_fname('feats_%s'%feat,'pdf'))
                    plt.tight_layout()
                    plt.close(10)
                plt.figure(4)
                for sel in range(tot_feats):
                    plt.subplot(tot_feats,1,sel+1)
                    emb_int = genEmb[paths[0],:]
                    emb_euc = genEmb[paths[1],:]
                    plt.plot(emb_int[:,sel],'r.-')
                    plt.plot(emb_euc[:,sel],'b.--')
                    plt.title('PC %d (explained var: %1.2f)'%(sel, explained_var[sel]))

                    # save
                    plt.figure(10)
                    plt.plot(emb_int[:,sel],'r.-')
                    plt.plot(emb_euc[:,sel],'b.--')
                    plt.savefig(save_fname('pcs_%d'%sel,'pdf'))
                    plt.tight_layout()
                    plt.close(10)

                with open(save_fname('expvar','txt'),'w') as f:
                    f.write(str(explained_var))

        plt.show()


def mainmain():
    dataset = 'kth'
    load_file = 'train_vars_'+dataset+'.bin'
    used_data_fname = 'used_data_'+dataset+'.bin'
    x_train0, x_test0, y_train0, y_test0 = pickle.load(open(used_data_fname,'r'))
    #print [ np.float32(x) for x in y_train0[0][[0,1,3,4]] ]
    train_vars_compressed, train_vars_std_compressed, train_vars_s_compressed = pickle.load(open(load_file,'r'))
    use_one_response = False
    if use_one_response:
        data = train_vars_compressed[0]
    else:
        data = np.concatenate(train_vars_compressed,axis=1)

    print 'data shape',data.shape
    emb_dim = 6
    Metric = LearnMetric(emb_dim=emb_dim,intrinsic_dim=3,sqeps=0.01)

    # discard first entry (fbm)
    discard_fbm_sample = False
    if discard_fbm_sample:
        data=data[1:,:]
        x_train0 = x_train0[1:]
        y_train0 = y_train0[1:]

    genEmb,H, explained_var = Metric.learnMetric(data)

    ## discrete geodesic k-means
    num_clusters = 6


    G, weights = Metric.calcDistance(data,genEmb,explained_var[:emb_dim],H)
    init_centroids = [0,10,20,30,40,50,60,70] # SET number of clusters here
    init_centroids=init_centroids[:num_clusters]

    centroids = dict(zip(weights,[ init_centroids,init_centroids ]))
    assignments = dict(zip(weights,[ [], [] ]))
    #plt.ion()
    for iter in range(5):
        print 'ITERATION', iter
        for i, weight in enumerate(weights):
            fig=plt.figure(i)
            ax=plt.subplot(3,4,iter+1)
            assignments[weight] = Metric.kmeans_assign(G,weight,centroids[weight],H.shape[2],genEmb,disp_i=True)
            centroids[weight] = Metric.kmeans_update(G,weight,assignments[weight],H.shape[2],genEmb,disp_i=None)
            extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            ax.set_title('%s: iter %d'%(weight,iter))
            fig.savefig('clust_%s_iter_%d.pdf'%(weight,iter), bbox_inches=extent)

    # save images by their assignments
    def mkdirr(x):
        try:
            mkdir(x)
        except:
            pass

    rmtree('clusters')
    mkdirr('clusters')
    if not discard_fbm_sample:
        y_train0[0][2]='fbm'
    subclasses = dict(zip(['woola','woolb','woolc','woold','fbm'],[0,1,2,3,4]))
    total_class_per_clust = {'int':[],'euc':[]}
    for w,assignment in assignments.iteritems():
        mkdirr('clusters/%s'%w[:3])
        class_per_clust = [ [] for i in range(num_clusters) ]
        for ind,(i,cluster) in enumerate(assignment.iteritems()):
            mkdirr('clusters/%s/%d'%(w[:3],i))
            for j in cluster:
                class_per_clust[ind].append(subclasses[y_train0[j][2]])
                fname = 'clusters/%s/%d/%d_true_%s.png'%(w[:3],i,j,y_train0[j][2])
                imsave(fname,x_train0[j])
        total_class_per_clust[w[:3]]=class_per_clust

    # see the cluster histograms
    stds = {'int':0,'euc':0}
    #correct_labeled={'int':0.0,'euc':0.0}
    cluster_mat = {'int':[],'euc':[]}
    for w,clust in total_class_per_clust.iteritems():
        print 'w',w
        for c in clust:
            hist1 = np.histogram(c,bins=[-0.5,0.5,1.5,2.5,3.5,4.5])[0]
            cluster_mat[w[:3]].append(hist1)
            #correct_labeled[w[:3]]+=np.max(hist1)
            #stds[w[:3]]+=np.std(hist1)
            #stds[w[:3]]+=np.std(c)
            #print c, np.var(hist1), hist1
        #correct_labeled[w[:3]] /= len(y_train0)
    #print 'intra-class std comparing true subclasses with clusters', stds
    cluster_mat['int']=np.array(cluster_mat['int'])
    cluster_mat['euc']=np.array(cluster_mat['euc'])
    #print 'cluster mat',cluster_mat
    cluster_mat_analysis(cluster_mat)
    plt.show()
    #raw_input('hit and key to end...')
    return
    ## show and save distances within the graph (from image A to B)

    Metric.exampleDistance(data,genEmb,explained_var[:emb_dim],H,
                           imgs=x_train0,
                           features=y_train0[:,[0,1,3,4]],
                           feature_names=['H','Kurtosis','MeanCoh','StdCoh'])

    return
    ## general display of the graph

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
    #plt.hold(True)

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

    #fig = mlab.figure(0)
    #fig.scene.anti_aliasing_frames=0
    #fig.scene.disable_render=True
    data2matlab = {'ellipses':[],'metric':[],'imgs':[],'features':[],'class':[]}
    for r, xx1,xx2,xx3,i  in zip(radi, x1, x2, x3, range(H.shape[2])):
        #plt.scatter(xx1,xx2,zs=xx3,c=r)
        A = H[:,:,i]
        center = [xx1,xx2,xx3]
        de = DrawEllipse(A,center,scale=0.1,color=r)
        data2matlab['metric'].append(A)
        data2matlab['ellipses'].append(de.getParams())

        #if not (i%30):
        data2matlab['imgs'].append(x_train0[i])
        #[h, kurt, name, np.mean(coh['logcoh']), np.std(coh['logcoh'])]
        data2matlab['features'].append([ np.float32(x) for x in y_train0[i][[0,1,3,4]] ])
        data2matlab['class'].append(y_train0[i][2])
        #ax0 = fig.add_axes([xx1,xx2,xx1+1,xx2+1])
        #ImageOverlay(cur,anchor=(xx1,xx2,xx3),size=(.001,.001,0))
        """ # plot images on 3d axes using matplotlib 
            rr = 0.01
            stepX, stepY = 2*2*rr / cur.shape[0], 2*rr / cur.shape[1]
            X1 = np.arange(-2*rr, 2*rr, stepX)
            Y1 = np.arange(-rr, rr, stepY)
            X1, Y1 = np.meshgrid(xx1+X1, xx3+Y1)
            #print np.min(cur),np.max(cur)
            ax.plot_surface(X1,xx2,Y1,rstride=10,cstride=10,facecolors=plt.cm.gray(cur/255.0),alpha=1.0)
                #imshow(imresize(cur,(5,5)), cmap=plt.cm.BrBG, interpolation='nearest', origin='lower', extent=[0,1,0,1])
            #fig.figimage(imresize(cur,(5,5)),xo=xx1,yo=xx2,cmap=plt.cm.gray)
            #ax0.imshow(imresize(cur,(0.1,0.1)))
            """
    sio.savemat('learn_metric_data.mat',data2matlab)
    print 'saved to mat file'
    #fig.scene.disable_render=False
    #mlab.show()

    #plt.xlabel('PC1')
    #plt.ylabel('PC2')
    #ax.set_zlabel('PC3')

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

    ax3 = plt.figure(1).add_subplot(111)
    ax3.plot(explained_var,'.-')
    ax3.set_xlabel('Eigenvalues')

    plt.show()


if __name__=='__main__':
    mainmain()

