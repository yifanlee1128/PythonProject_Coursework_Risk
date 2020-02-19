import qrbook_funcs as qf
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import math
#FRED codes for US Treasury constant maturity rates
seriesnames=['DGS1MO','DGS3MO','DGS6MO','DGS1','DGS2','DGS3','DGS5','DGS7','DGS10','DGS20','DGS30']
maturities=qf.TenorsFromNames(seriesnames)
dates,prices=qf.GetFREDMatrix(seriesnames,startdate='2018-12-31',enddate='2018-12-31')

prices=prices[0]
x=np.linspace(1,360,360)
xval=[1,3,6,12,24,36,60,84,120,240,360]
price=np.interp(x,xval,prices)
def mad(x):
    maturity=np.linspace(1,360,360).tolist()
    v=[x[0] + x[1] * x[3] / i * (1 - math.exp(-i / x[3])) + x[2] * x[3] / i * (
                    1 - math.exp(-i / x[3]) * (1 + i / x[3]))for i in maturity]
    diff=[]
    for a,b in zip(price,v):
        diff.append(a-b)
    diff=np.abs(diff-np.mean(diff))
    return np.sum(diff)/len(diff)
beta0_start=3.0
beta1_start=0.58
beta2_start=2.8
tau_start=100.0
x0 = np.array([beta0_start,beta1_start,beta2_start,tau_start])
res = minimize(mad, x0, method='nelder-mead',
                options={'xtol': 1e-8, 'disp': True, 'maxiter': 1000})
print(res.x)