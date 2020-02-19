import pandas as pd
import qrbook_funcs as qf
import numpy as np
import matplotlib.pyplot as plt
#FRED codes for US Treasury constant maturity rates
seriesnames=['DGS1MO','DGS3MO','DGS6MO','DGS1','DGS2','DGS3','DGS5','DGS7','DGS10','DGS20','DGS30']
maturities=qf.TenorsFromNames(seriesnames)
dates,prices=qf.GetFREDMatrix(seriesnames,startdate='2018-01-02',enddate='2018-12-31')

#remove no-data periods
nobs, t = len(dates), 0
while t<nobs:
    if all(np.isnan(prices[t])):
        del prices[t]
        del dates[t]
        nobs -= 1
    else:
        t += 1

diff=[[0]*len(seriesnames)]*len(dates)
print(diff[0][0])
print(diff)
for i in range(len(prices)-1):
    for j in range(len(prices[0])):
        #print(i,j)
        diff[i][j]=prices[i+1][j]-prices[i][j]
print(diff)
diffs=np.mat(diff)
print(diffs)

