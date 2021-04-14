import numpy as np

a = np.array([1,2,3])
print(a)
print("a.shape:", a.shape)
b = np.array([4,5,6])
print(b)
print("b.shape:", b.shape)

d = np.array([a, b])
print(d)
print("d.shape", d.shape)

e = np.transpose(d)
print(e)
print("e.shape", e.shape)
