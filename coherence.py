import numpy as np
from scipy.misc import imread
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from numpy.linalg import eig

def calc_image_derivatives(im):
    Fx=1/2.0*np.array([[0, 0, 0,],
            [-1, 0, 1],
            [0, 0, 0]],dtype=np.float32)
    Fy=-Fx.T
    Ix = convolve2d(im,Fx,mode='same',boundary='symm')
    Iy = convolve2d(im,Fy,mode='same',boundary='symm')
    def zeroBound(I):
        I[:,0]=0
        I[:,-1]=0
        I[0,:]=0
        I[-1,:]=0
        return I
    Ix=zeroBound(Ix)
    Iy=zeroBound(Iy)
    return Ix, Iy


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


def compute_structural_tensor(Ix,Iy,rho):
    """
     Input:
     Ix,Iy - the horizontal and vertical derivatives of the image
     rho - the parameter for the width of the gaussian filter that is used for computing the average structural tensor

     Output:
     J11,J12,J22 - the average structural tensor (remember that J21=J12)

     compute the structural tensor for each pixel in the image
    """
    size_win=1+4*rho

    J11 = Ix**2
    J12 = Ix*Iy
    J22 = Iy**2

    h_gauss=gauss2D_filter(shape=(size_win,size_win),
                           sigma=np.sqrt(rho),one_D=False)

    J11 = convolve2d(J11,h_gauss,'same','symm')
    J12 = convolve2d(J12,h_gauss,'same','symm')
    J22 = convolve2d(J22,h_gauss,'same','symm')
    return J11,J12,J22

def compute_coherence_and_orientations(J11, J12, J22):
    """
     Input:
     J11,J12,J22 - the average structural tensor (remember that J21=J12)

     Output:
     coh - a matrix containing for each pixel the cohernce value
     V1 - a 2-layer matrix containing for each pixel a vector that is directed towards the edge orientation
     V2 - a 2-layer matrix containing for each pixel a vector that is directed towards the coherence orientation

     compute the coherence value and the coherence and edge orientations for each pixel in the image
    """
    delta=np.sqrt(((J11-J22)**2+4*J12**2))
    coh = delta**2
    u1=0.5*(J11+J22+delta)
    cond_for_V = np.abs(J12)>=1.0
    V11 = np.zeros_like(u1)
    V11[cond_for_V] = np.sqrt(1.0/(1+((1.0*u1[cond_for_V]-J11[cond_for_V])/J12[cond_for_V])**2))
    V1 = np.zeros((u1.shape[0],u1.shape[1],2))
    V2 = np.zeros_like(V1)
    V1[:,:,0] = V11
    V12 = np.zeros_like(u1)
    cond = 2*(((u1[cond_for_V]-J11[cond_for_V])/J12[cond_for_V])>0)

    V12[cond_for_V] = (-1.0+cond)*np.sqrt(1-V11[cond_for_V]**2)

    V1[:,:,1] = V12
    V2[:,:,0] = V1[:,:,1]
    V2[:,:,1] = -V1[:,:,0]
    [row,col]=np.where(~cond_for_V)
    for i in range(len(row)):
        r, c = row[i], col[i]
        J=np.array([[ J11[r,c],J12[r,c]],
            [J12[r,c],J22[r,c]]])
        Q, V = eig(J)
        if Q[0]<Q[1]:
            V = V[:,::-1]
        V1[r,c,:]=V[:,0]
        V2[r,c,:]=V[:,1]

    return coh,V1,V2

def coherence(im,rho=3):
    im = 1.0*im-np.min(im)
    im = im/np.max(im)*256.0

    Ix, Iy = calc_image_derivatives(im)
    J11, J12, J22 = compute_structural_tensor(Ix,Iy,rho)
    coh, V1, V2 = compute_coherence_and_orientations(J11,J12,J22)
    logcoh = np.log(1+coh)
    coh_med = np.median(logcoh.flatten())
    angles = np.arctan(1.0*V1[:,:,1]/V1[:,:,0])
    out = {'coh':coh,'logcoh':logcoh,'V1':V1,'V2':V2,'coh_med':coh_med,'angles':angles}
    return out#coh,logcoh,V1,V2,coh_med,angles

if __name__=='__main__':
    im = imread('mnist.png')
    im = im[:256,:256,:]

    im = np.mean(im,axis=2)
    out = coherence(im)
    c = out['coh']
    print np.mean(out['logcoh']),np.std(out['logcoh'])
    plt.subplot(2,1,1)
    plt.imshow(im,cmap=plt.cm.gray)
    plt.subplot(2,1,2)
    plt.imshow(c,cmap=plt.cm.gray)
    plt.show()
