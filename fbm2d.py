import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
def hurst2d(ts,max_tau=30):

    """Returns the Hurst Exponent of the time series vector ts"""

    # extension of estimate to 2D fBm is performed via taking increments along main axes

    # Create the range of lag values
    lags = range(2, max_tau)

    # Calculate the array of the variances of the lagged differences
    # Here it calculates the variances, but why it uses
    # standard deviation and then make a root of it?
    taus=[]
    for lag in lags:
        v1 = np.subtract(ts[lag:,:], ts[:-lag,:])
        v2 = np.subtract(ts[:,lag:], ts[:,:-lag])

        v = np.vstack([v1.flatten(),v2.flatten()])
        taus.append(np.sqrt(np.std(v)))
    tau=np.array(taus)
    #tau_y = [np.sqrt(np.std()) for lag in lags]
    #tau = 0.5*(np.array(tau_x)+np.array(tau_y)) # not sure if the best way

    # Use a linear fit to estimate the Hurst Exponent
    p0, p1, rval, _, _ = linregress(np.log(lags),np.log(tau))
    #poly, _, _, _, rcond = np.polyfit(np.log(lags), np.log(tau), 1, full=True)

    # Return the Hurst exponent from the polyfit output
    return p0*2.0, rval**2

def strfbm(x,y,H):
    # structure function for fBm
    return np.power(x**2+y**2,H)

def symcori(S):
    #print S.shape
    # Symmetrization of a periodic 2D correlation field
    N = S.shape[1]
    if S.shape[0]<>N/2+1:
        print 'ERROR in symcori r dimensions'
        return None
    else:
        out = np.copy(S)
        ind = np.arange(N/2,N-1)
        out[0,ind] = 0.5*(np.conj(out[0,N-ind])+out[0,ind])
        out[0,N-ind] = np.conj(out[0,ind])
        out[N/2,ind] = 0.5*(np.conj(out[N/2,N-ind])+out[N/2,ind])
        out[N/2,N-ind] = np.conj(out[N/2,ind])

        #print ind
        #print N-ind
        out = np.vstack([out, np.zeros([N/2-1,N])])
        out[ind+1,0] = np.conj(S[N-ind-1,0])
        ind2 = np.arange(N-1)
        I1,J1 = np.ix_(ind+1,ind2+1)
        I2,J2 = np.ix_(N-ind-1,N-ind2-1)
        #print I2, J2
        #print out.shape, ind, ind2+1
        #print out[ind,ind2+1].shape
        #print S[N-ind,N-ind2].shape
        #out[ind,ind2+1] = np.conj(S[N-ind,N-ind2])
        out[I1,J1] = np.conj(S[I2,J2])
        """
        plt.imshow(out,interpolation='none')
        #plt.imshow(S,interpolation='none')
        plt.colorbar()
        plt.show()
        #"""
        return out

def synth2(N=128,H=0.5,W=None,Wx=None,Wy=None):
    # derived from FracLab MATLAB library
    # Original author: B. Pesquet-Popescu, 1998
    # INRIA 2009
    # adapted to Python by Ido Zachevsky

    # N - size of image axis, H - Hurst parameter between 0 and 1
    # W, Wx, Wy - initial random noise for generation
    # output: 2D fBm field with size NxN and Hurst parameter H
    # This is an approximation that can be made efficiently using Fourier synthesis
    # which is much more efficient than direct covariance calculation.

    # 1. Initial noise generation
    M = 2*N
    M2 = [M,M]
    if W is None:
        noise1 = np.random.randn(M,M)
        noise2 = np.random.randn(M)/np.sqrt(M) # column
        noise3 = np.random.randn(M)/np.sqrt(M) # row
        W = np.fft.fft2(noise1)
        Wx = np.fft.fft(noise2)
        Wy = np.fft.fft(noise3)

    # 2. Increment correlation

    # coordinate system
    ind = np.arange(M/2+1)
    ind1 = np.hstack([ind, np.arange(-M/2+1,0)])
    Ind1,Ind2 = np.meshgrid(ind,ind1)
    Ind1 = np.transpose(Ind1)
    Ind2 = np.transpose(Ind2)

    # increment corr
    fun = lambda x,y: strfbm(x,y,H)

    term1 = 2*( fun(Ind1+1,Ind2)+fun(Ind1-1,Ind2)+
                fun(Ind1,Ind2+1)+fun(Ind1,Ind2-1))
    term2 = -(fun(Ind1+1,Ind2+1)+fun(Ind1+1,Ind2-1)+
            fun(Ind1-1,Ind2+1)+fun(Ind1-1,Ind2-1))
    term3 = -4*fun(Ind1,Ind2)

    r2 = 0.5*(term1+term2+term3)
    r2 = symcori(r2)

    S2 = np.real(np.fft.fft2(r2))

    S2[S2<0]=0
    S2[0,:] = 0
    S2[:,0] = 0

    I2 = np.sqrt(S2)*W

    i2 = np.real(np.fft.ifft2(I2))
    #print 'i2',i2.shape
    I1,J1 = np.ix_(range(N),range(N))
    i2 = i2[I1,J1]

    # step 8

    rx = 0.5*( fun(Ind1+1,Ind2) + fun(Ind1-1,Ind2) -2*fun(Ind1,Ind2) )
    ry = 0.5*( fun(Ind1,Ind2+1) + fun(Ind1,Ind2-1) -2*fun(Ind1,Ind2) )

    rx = symcori(rx)
    ry = symcori(ry)

    Sx = np.transpose(np.real(np.fft.fft(np.sum(rx,1))))
    Sy = np.real(np.fft.fft(np.sum(ry,0)))

    Sx[Sx<0]=0
    Sy[Sy<0]=0

    # step 11

    Ix = np.zeros(M2,dtype='complex')
    Iy = np.zeros(M2,dtype='complex')
    indis = np.arange(M-1)

    inner1 = np.pi*(1+indis)/M
    term1 = np.outer(np.ones(M-1), np.exp(-1j*inner1)/np.sin(inner1))
    I1,J1=np.ix_(indis,indis)
    Ix[I1,J1] = -1j * I2[I1,J1]/2 * term1
    Iy[I1,J1] = -1j * I2[I1,J1]/2 * np.transpose(term1)

    Ix[:,0] = M*np.sqrt(Sx)*Wx
    Iy[0,:] = M*np.sqrt(Sy)*Wy

    # 12

    ix = np.fft.ifft(np.mean(Ix,1))
    ix = np.real(ix[0:N])

    iy = np.fft.ifft(np.mean(Iy,0))
    iy = np.real(iy[0:N])

    fBm = np.zeros([N,N])
    fBm[0,0]=0
    for mx in range(1,N):
        fBm[mx,0] = fBm[mx-1,0] + ix[mx-1]

    for my in range(1,N):
        fBm[0,my] = fBm[0,my-1] + iy[my-1]


    for mx in range(1,N):
        for my in range(1,N):
            fBm[mx,my] = fBm[mx,my-1] + fBm[mx-1,my] - fBm[mx-1,my-1] + i2[mx-1,my-1]

    return fBm


if __name__=='__main__':
    #fbm = synth2(N=8,H=0.5)
    fbm = synth2(N=32,H=0.4)

    #r2 = symcori(np.ones([17,32]))
    #plt.imshow(r2,interpolation='none')
    plt.imshow(fbm,interpolation='none',cmap='gray')
    plt.show()
