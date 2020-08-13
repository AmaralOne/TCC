import UtilsM3
import numpy
import pandas as pd
    
U_m3 = UtilsM3.UtilsM3()
index = U_m3.listarIndex()
ts = U_m3.buildM3DataFrame("N1681")

dataset = ts.copy()

print('ts: ',ts[0])
print('dataset: ',dataset[0])

dataset[0] = -1

print('ts: ',ts[0])
print('dataset: ',dataset[0])