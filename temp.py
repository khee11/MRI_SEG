#%%
import pandas as pd
import numpy as np
import os

PATH = '/data/HWKIM/unconverted/BraTS2023_Training'
df = []
for i in os.listdir(PATH):
    one_row = []
    path = os.path.join(PATH, i)
    one_row.append(i)
    one_row.append(path)
    df.append(one_row)
df = pd.DataFrame(df, columns = ['id', 'path'])
df.to_csv('')
#%%