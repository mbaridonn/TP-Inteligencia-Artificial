import numpy as np
import pandas

a = np.array([[1],[2],[3]])

print(a)


df = pandas.DataFrame({'Data': a[:,0]})
print(df)

df.to_excel("output.xlsx")