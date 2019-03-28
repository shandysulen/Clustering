import numpy as np

c1 = np.concatenate((np.random.rand(50, 1), np.random.rand(50, 1) * 3), axis=1)
c2 = np.concatenate((np.random.rand(50, 1)+ 1.5, np.random.rand(50, 1) * 3), axis=1)
d = np.concatenate((c1, c2), axis=0)
print(d)