import numpy as np
import pandas as pd
import qrbook_funcs as qf
#Get 3 currencies until the end of
#previous year. Form sample covariance matrix
#and do simple efficient frontier calculations

lastday=qf.LastYearEnd()
#Swiss franc, pound sterling, Japanese Yen
seriesnames=['DEXSZUS','DEXUSUK','DEXJPUS']
cdates,ratematrix=qf.GetFREDMatrix(seriesnames,enddate=lastday)

#Convert levels to log-returns
#First take logs of the currency levels
#Currency exchange rates are usually expressed in the direction
#that will make the rate > 1
#Swissie and yen are in currency/dollar, but
#pounds is in dollar/currency. Reverse signs
#so everything is in dollar/currency

#Do each currency separately to account for separate missing data patterns
#dlgs is a list of lists of length 3 corresponding to the 3 currencies
#The value in dlgs is nan if there is missing data for the present or previous day's observation
#Otherwise it is the log of today/yesterday
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

wmkt=np.array([.05,.15,.8])
mumkt=np.matmul(wmkt,m.T)
varmkt=np.matmul(np.matmul(wmkt,c),wmkt.T)
betavec=np.matmul(c,wmkt)/varmkt

rfrate=10**(-5)
rfvec=[rfrate]*3
mucapm=np.multiply(10000,rfvec+(mumkt-rfrate)*betavec)

view=np.array([0,1,-1])
pview=.00002
gamma=np.matrix([.0001])
sweight=3.834623  # The value of s we need

v1=np.matmul(np.matrix(view).T,np.linalg.inv(gamma))
vgvmtrx=np.matmul(v1,np.matrix(view))
ci=np.linalg.inv(c)
m1inv=np.linalg.inv(ci/sweight+vgvmtrx)
print("The value of the matrix is below:")
print(str((ci/sweight+vgvmtrx)/10000))

cimcs=np.matmul(ci,mucapm/sweight)*10**(-4)
m2=cimcs+v1.T*pview
print("The value of the vector is: "+str(m2))
mufinal=np.matmul(m1inv,m2.T)*10000
print("My new estimated mean vector is: "+str(mufinal.T))