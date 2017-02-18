import numpy as np

x = ([[2,4,3,5,6,7,4,5,6,3,4,6],
     [2,3,4,5,4,2,3,4,5,4,3,2],
     [7,8,7,6,5,6,7,6,5,6,7,6]])
x = np.array(x)
print(x.shape[0], x.shape[1])

y = np.reshape(x, (x.shape[0], 1, x.shape[1]))

print(y)