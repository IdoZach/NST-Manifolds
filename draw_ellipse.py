import numpy as np
import numpy.linalg as linalg
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# https://stackoverflow.com/questions/7819498/plotting-ellipsoid-with-matplotlib

class DrawEllipse():
    def __init__(self,A=None,center=None,scale=1.0,color='b'):
        # your ellispsoid and center in matrix form
        if A is None:
            A = np.array([[1,0,0],[0,2,0],[0,0,2]])
        if center is None:
            center = [0,0,0]

        # find the rotation matrix and radii of the axes
        U, s, rotation = linalg.svd(A)
        radii = 1.0/np.sqrt(s) * scale
        self.color = color
        # now carry on with EOL's answer
        u = np.linspace(0.0, 2.0 * np.pi, 100)
        v = np.linspace(0.0, np.pi, 100)
        x = radii[0] * np.outer(np.cos(u), np.sin(v))
        y = radii[1] * np.outer(np.sin(u), np.sin(v))
        z = radii[2] * np.outer(np.ones_like(u), np.cos(v))
        for i in range(len(x)):
            for j in range(len(x)):
                [x[i,j],y[i,j],z[i,j]] = np.dot([x[i,j],y[i,j],z[i,j]], rotation) + center
        self.xyz = [x,y,z]

    def get_xyz(self):
        return self.xyz

    def plot(self,ax=None):
        x,y,z = self.xyz
        if ax is None:
            show = True
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
        else:
            show = False

        ax.plot_wireframe(x, y, z,  rstride=16, cstride=16, color=self.color, linewidth=0.4, alpha=0.5)

        if show:
            plt.show()
            plt.close(fig)
            del fig

if __name__ == '__main__':
    de = DrawEllipse()

