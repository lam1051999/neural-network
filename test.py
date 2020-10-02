import numpy as np
import sigmoid

a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
b = np.array([[1/2, 2, 3], [4, 5, 6], [7, 8, 9]])
x = a == b
print(x[x == True].size)
