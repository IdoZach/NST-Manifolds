import matplotlib.pyplot as plt
import numpy as np

x = np.array([ [1,2,3],[4,5,6],[7,8,9]])
y = x + np.random.randn(3,3)

plt.imshow(x)
plt.show()
plt.pause(1)
plt.imshow(y)
plt.show()
plt.pause(1)
print('done')


