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
tenornames=['1mo','3mo','6mo','1yr','2yr','3yr',
        '4yr','5yr','7yr','10yr','30yr']
tenornumbers=range(len(tenornames))

over_day_change=[]
for a,b in zip(prices[1:],prices[:-1]):
    over_day_change.append([x - y for x, y in zip(a, b)])

for_cov=np.matrix(over_day_change).transpose()
cov_matrix=np.cov(for_cov)
w,v=np.linalg.eig(cov_matrix)

PC1=v.transpose()[0]
PC2=v.transpose()[1]
PC3=v.transpose()[2]

plt.plot(tenornumbers ,PC1, label='PC1')
plt.plot(tenornumbers ,PC2, label='PC2')
plt.plot(tenornumbers ,PC3, label='PC3')
plt.title('UST Curve Principal Components')
plt.xlabel('Tenor')
plt.ylabel('Level')
plt.legend()
plt.xticks(tenornumbers, tenornames)
plt.grid(True)
plt.savefig('UST Curve Principal Components.pdf')
plt.show()

percentage=w[:3].sum()/w.sum()
print("percentage of the trace from the first 3 PC is: "+str(percentage))