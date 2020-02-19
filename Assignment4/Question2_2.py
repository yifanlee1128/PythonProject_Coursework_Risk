import qrbook_funcs as qf
import numpy as np
import matplotlib.pyplot as plt

#FRED codes for US Treasury constant maturity rates
seriesnames=['DGS1MO','DGS3MO','DGS6MO','DGS1','DGS2','DGS3','DGS5','DGS7','DGS10','DGS20','DGS30']
maturities=qf.TenorsFromNames(seriesnames)
dates,prices=qf.GetFREDMatrix(seriesnames,startdate='2018-01-02',enddate='2018-12-31')
# remove no-data periods
nobs, t = len(dates), 0
while t < nobs:
    if all(np.isnan(prices[t])):
        del prices[t]
        del dates[t]
        nobs -= 1
    else:
        t += 1


# computing the day-over-day change
change = np.zeros((len(prices)-1,11))
for k in range(0,len(prices)-1):
    for l in range(0,11):
        change[k][l] = prices[k+1][l] - prices[k][l]

# compute the cov matrix
cov = np.cov(change.T)

#compute the principle components
value, vector = np.linalg.eig(cov)


# compute the percentage
percentage = sum(value[0:3])/sum(value) # value[0:3] is actually value[0],value[1] and value[2]
print(percentage)

# draw the plot

tenornames = seriesnames
tenornumbers=range(len(tenornames))
vector = vector.T
pc1=vector[0]
pc2=vector[1]
pc3=vector[2]

plt.plot(tenornumbers, pc1, label='PC1')
plt.plot(tenornumbers, pc2, label='PC2')
plt.plot(tenornumbers, pc3, label='PC3')

## Configure the graph
plt.title(' Curve Principal Components')
plt.xlabel('Tenor')
plt.ylabel('Level')
plt.legend()
plt.xticks(tenornumbers, tenornames)
plt.grid(True)
plt.show();
