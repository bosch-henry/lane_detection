import pandas as pd
import numpy as np

cont_list = [{"name" : "gx", "age" : 31}, {"name" : "zy", "age" : 30}, {"name" : "gxl", "age" : 60}]

df = pd.DataFrame(cont_list, columns = ["name", "age"])

print(df)

#df.to_csv("./famaily.csv", index = False)

data = pd.read_csv("famaily.csv")

print(data)
print(data.columns)
print(data.shape)
print(data.loc[1:2])
print(data.loc[ : , ['age', 'work']])

data_array = np.array(data)

print(data_array)

print(data_array[ : , 1])
print(data_array[ : , 1].shape)