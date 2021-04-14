import numpy as np
import pandas as pd
from util.file_util import *

arr1 = np.arange(100).reshape(50,2)
print("arr1 = \n", arr1)
print(arr1.shape)

data1 = pd.DataFrame(arr1, columns=["X", "Y"])
print("data1 = \n", data1)

data1.to_csv('data1.csv', index=False)

lst = [1,2,3,4]
arr2 = np.array(lst).reshape(-1,1)
print("arr2 = \n", arr2)
print(arr2.shape)
data2 = pd.DataFrame(arr2, columns=["coef"])
print("data2 = \n", data2)

data2.to_csv('data2.csv', index=False)

