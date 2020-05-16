import pandas as pd
import numpy as np

df=pd.read_csv('CISI.REL', sep=' ',header=None)
for i in range(1, 113):
    if (not np.any(df.loc[:,0] == i)):
        df2 = pd.DataFrame([[i, 0, 1166, 0]])
        df = df.append(df2,ignore_index=True)
df = df.sort_values(by=[0])
df.to_csv('CISI.REL.NEW', sep=' ', index = False, header=None)