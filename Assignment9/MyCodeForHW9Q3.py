import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import qrbook_funcs as qf
from scipy import stats
#Get 3 currencies until the end of
#previous year. Form sample covariance matrix
#and do simple efficient frontier calculations

lastday=qf.LastYearEnd()
#Swiss franc, pound sterling, Japanese Yen
seriesnames=['DEXSZUS','DEXUSUK','DEXJPUS']
cdates,ratematrix=qf.GetFREDMatrix(seriesnames,enddate=lastday)

multipliers=[-1,1,-1]
dlgs=[]
for i in range(len(multipliers)):
    lgrates=[]
    previous=-1
    for t in range(len(ratematrix)):
        if pd.isna(ratematrix[t][i]) or ratematrix[t][i]<=0:
            lgrates.append(np.nan)    #Append a nan
        else:
            if previous < 0:    #This is the first data point
                lgrates.append(np.nan)
            else:
                lgrates.append(np.log(ratematrix[t][i]/previous)*multipliers[i])
            previous=ratematrix[t][i]
    dlgs.append(lgrates)

#dlgs is the transpose of what we want - flip it
dlgs=np.transpose(dlgs)

#Delete any time periods that don't have data
lgdates=[]
difflgs=[]
for t in range(len(dlgs)):
    if all(pd.notna(dlgs[t])):
        #include this time period
        difflgs.append(dlgs[t])
        lgdates.append(cdates[t])

data=difflgs[-1000:]
portfolio=[ np.sum(i)/3 for i in data]
portfolio_mean=np.mean(portfolio)
portfolio_std=np.std(portfolio)
portfolio_kurt=stats.kurtosis(portfolio,fisher=True)
print(portfolio_mean,portfolio_std,portfolio_kurt)
w1_list=[0.05,0.1,0.15,0.2];

#cal_r1=lambda w,k:(-w*(1-w)*(k+3)+np.sqrt(3*k*w*(1-w)))/(w*((k+3)*w-3))

cal_r=lambda w,k:(-w*(1-w)*(k+3)-np.sqrt(3*k*w*(1-w)))/(w*((k+3)*w-3))
r_list=[cal_r(w1,portfolio_kurt) for w1 in w1_list]

cal_sigma2=lambda sigma,w1,r:np.sqrt(sigma**2/(w1*r+(1-w1)))
sigma2_list=[cal_sigma2(portfolio_std,w1,r) for w1,r in zip(w1_list,r_list)]

print(r_list)
print(sigma2_list)
