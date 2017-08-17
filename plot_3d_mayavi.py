from mayavi import mlab
import matplotlib.image as mpimg
import numpy as np
import numpy.linalg as linalg

# https://stackoverflow.com/questions/7819498/plotting-ellipsoid-with-matplotlib
class DrawEllipse():
    def __init__(self,A=None,center=None,scale=1.0,color='b'):
        # your ellispsoid and center in matrix form
        if A is None:
            A = np.array([[1,0,0],[0,2,0],[0,0,2]])
        if center is None:
            center = [0,0,0]

        # find the rotation matrix and radii of the axes
        U, s, self.rotation = linalg.svd(A)
        self.radii = 1.0/np.sqrt(s) * scale
        self.color = color
        self.center = center

    def getParams(self):
        return self.center, self.radii

    def plot(self):
        # now carry on with EOL's answer
        res = 20 # 100
        u = np.linspace(0.0, 2.0 * np.pi, res)
        v = np.linspace(0.0, np.pi, res)
        x = self.radii[0] * np.outer(np.cos(u), np.sin(v))
        y = self.radii[1] * np.outer(np.sin(u), np.sin(v))
        z = self.radii[2] * np.outer(np.ones_like(u), np.cos(v))
        for i in range(len(x)):
            for j in range(len(x)):
                [x[i,j],y[i,j],z[i,j]] = np.dot([x[i,j],y[i,j],z[i,j]], self.rotation) + self.center
        self.xyz = [x,y,z]
        x,y,z = self.xyz
        mlab.mesh(x,y,z,representation='wireframe',resolution=1,line_width=0.1)#,representation='wireframe')

        #ax.plot_wireframe(x, y, z,  rstride=16, cstride=16, color=self.color, linewidth=0.4, alpha=0.5)

class ImageOverlay():
    def __init__(self,cur=None,anchor=(1.,1.,1.),size=(0.01,0.01,0.01)):
        self.cur = mpimg.imread('mnist_1000.png')[:,:,0] if cur is None else cur
        self.anchor = anchor
        self.size = size
        self.draw()
    def draw(self):
        sz = self.size
        anc = self.anchor
        mlab.imshow(self.cur,
                    extent = [anc[0], sz[0],
                              anc[1], sz[1],
                              anc[2], sz[2]],
                    colormap='gist_earth')
        #ax.plot_surface(X1,y,Y1,rstride=10,cstride=10,facecolors=plt.cm.gray(self.cur/255.0),alpha=1.0)


if __name__ == '__main__':
    fig = mlab.figure(0)
    fig.scene.anti_aliasing_frames=0
    #mlab.anti_aliasing_frames=0
    fig.scene.disable_render=True
    ImageOverlay()
    DrawEllipse()
    fig.scene.disable_render=False
    mlab.show()
