import matplotlib.pyplot as plt
import matplotlib
import numpy as np

h=[]
hh,=plt.plot(np.array([1,2,3,4]),np.array([2,3,4,5]))
h.append(hh)
hh,=plt.plot(np.stack([[1,2,3,4],[2,3,4,5]]),np.stack([[2,3,4,6],[1,2,3,4]]))
print hh
h.append(hh)
plt.legend(h,['a','b'])
plt.show()

print matplotlib.__version__
