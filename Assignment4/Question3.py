import qrbook_funcs as qf
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
#FRED codes for US Treasury constant maturity rates
seriesnames=['DGS1MO','DGS3MO','DGS6MO','DGS1','DGS2','DGS3','DGS5','DGS7','DGS10','DGS20','DGS30']
maturities=qf.TenorsFromNames(seriesnames)
dates,prices=qf.GetFREDMatrix(seriesnames,startdate='2018-12-31',enddate='2018-12-31')

prices=prices[0]
x=np.linspace(1,360,360)
xval=[1,3,6,12,24,36,60,84,120,240,360]
price=np.interp(x,xval,prices)
def mad(x):
    maturity=np.linspace(1/12,30,360).tolist()
    v=[(x[0] + x[1] * x[3] / i * (1 - np.exp(-i / x[3])) + x[2] * x[3] / i * (
                    1 - np.exp(-i / x[3]) * (1 + i / x[3])))for i in maturity]
    diff=[]
    for a, b in zip(price, v):
        diff.append(np.abs(a - b))
    diff = np.abs(diff - np.mean(diff))
    return np.sum(diff) / len(diff)

# The result of last optimization, they all converge to the stable values
beta0_start = 3.59141179
beta1_start = -1.14777565
beta2_start = 2.14813936
tau_start = 74.16395268

x0 = np.array([beta0_start,beta1_start,beta2_start,tau_start])
res = minimize(mad, x0, method='nelder-mead',
                options={'xtol': 1e-10, 'disp': True, 'maxiter': 2000})

print(res.x)
# read the data and draw the result
maturity=np.linspace(1/12,30,360).tolist()
v=[(res.x[0] + res.x[1] * res.x[3] / i * (1 - np.exp(-i / res.x[3])) + res.x[2] * res.x[3] / i * (
                    1 - np.exp(-i / res.x[3]) * (1 + i / res.x[3])))for i in maturity]

plt.plot(x,v,label = 'Nelson-Siegel curve')
plt.plot(x, price, label = 'straight-line curve')
plt.title('Compare the Nelson-Siegel curve to straight-line curve')
plt.xlabel('Tensor')
plt.ylabel('Rate')
plt.legend()
plt.savefig('Comparison.pdf')
plt.show()