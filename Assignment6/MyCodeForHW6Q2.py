import pandas as pd
import qrbook_funcs as qf
import numpy as np
#Get 3 currencies until the end of
#previous year. Form sample covariance matrix
#and do simple efficient frontier calculations

lastday=qf.LastYearEnd()
#Swiss franc, pound sterling, Japanese Yen
seriesnames=['DEXSZUS','DEXUSUK','DEXJPUS']
cdates,ratematrix=qf.GetFREDMatrix(seriesnames,enddate="2017-12-29")
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

#Mean vector and covariance matrix are inputs to efficient frontier calculations
d=np.array(difflgs)
m=np.mean(d,axis=0)
c=np.cov(d.T)

prev_sig=np.sqrt(np.diag(np.diag(c)))
prev_sig_inverse=np.linalg.inv(prev_sig)
prev_r_matrix=np.matmul(np.matmul(prev_sig_inverse,c),prev_sig_inverse)

prev_dim=3
prev_avg_corr=(np.sum(prev_r_matrix)-prev_dim)/(prev_dim**2-prev_dim)

C_rho=np.matmul(np.matmul(prev_sig,(np.ones((prev_dim,prev_dim))*prev_avg_corr+np.identity(prev_dim)*(1-prev_avg_corr))),prev_sig,)

C_new=1/3*C_rho+(1-1/3)*c

ci=np.linalg.inv(c)
uciu=np.sum(ci)
u=[1]*3
portfolio=np.matmul(ci,u)/uciu
print("minimum variance portfolio w1 is: "+str(portfolio))

ci_new=np.linalg.inv(C_new)
uciu_new=np.sum(ci_new)
u=[1]*3
portfolio_new=np.matmul(ci_new,u)/uciu_new
print("minimum variance portfolio w2 is:"+str(portfolio_new))

cdates,ratematrix=qf.GetFREDMatrix(seriesnames,startdate='2018-01-01',enddate="2018-12-31")
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

#Mean vector and covariance matrix are inputs to efficient frontier calculations
d=np.array(difflgs)
m=np.mean(d,axis=0)
c_2018=np.cov(d.T)

variance_old=np.matmul(portfolio,np.matmul(c_2018,portfolio))
print("variance 1 is: "+str(variance_old*1e4))

variance_new=np.matmul(portfolio_new,np.matmul(c_2018,portfolio_new))
print("variance 2 is: "+str(variance_new*1e4))
