import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import qrbook_funcs as qf
from scipy import stats


#Swiss franc, pound sterling, Japanese Yen
seriesnames=['DEXUSUK']
cdates,ratematrix=qf.GetFREDMatrix(seriesnames,startdate='2014-12-31',enddate='2017-12-29')

multipliers=[1]
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

initparams=[.05,.94,.001]
a,b,c=qf.Garch11Fit(initparams,difflgs)

print("a=%.3f" % a)
print("b=%.3f" % b)
print("c=%.3f" % c)

cdates1,ratematrix1=qf.GetFREDMatrix(seriesnames,startdate='2017-12-29',enddate='2018-12-31')
dlgs1=[]
for i in range(len(multipliers)):
    lgrates1=[]
    previous=-1
    for t in range(len(ratematrix1)):
        if pd.isna(ratematrix1[t][i]) or ratematrix1[t][i]<=0:
            lgrates1.append(np.nan)    #Append a nan
        else:
            if previous < 0:    #This is the first data point
                lgrates1.append(np.nan)
            else:
                lgrates1.append(np.log(ratematrix1[t][i]/previous)*multipliers[i])
            previous=ratematrix1[t][i]
    dlgs1.append(lgrates1)

#dlgs is the transpose of what we want - flip it
dlgs1=np.transpose(dlgs1)

#Delete any time periods that don't have data
lgdates1=[]
difflgs1=[]
for t in range(len(dlgs1)):
    if all(pd.notna(dlgs1[t])):
        #include this time period
        difflgs1.append(dlgs1[t])
        lgdates1.append(cdates1[t])

t=len(difflgs1)
minimal=10**(-20)
stdgarch=np.zeros(t)
stdgarch[0]=np.std(difflgs1)
degarched=np.zeros(t)   #series to hold de-garched series y[t]/sigma[t]
degarched[0]=difflgs1[0]/stdgarch[0]
#Compute GARCH(1,1) stddev's from data given parameters
for i in range(1,t):
    #Note offset - i-1 observation of data
    #is used for i estimate of std deviation
    previous=stdgarch[i-1]**2
    var=c+b*previous+\
        a*(difflgs1[i-1])**2
    stdgarch[i]=np.sqrt(var)
    degarched[i]=difflgs1[i]/stdgarch[i]

kurt_degarch=stats.kurtosis(degarched,fisher=True)
std_degarch=np.std(degarched)
print("--------------------------")
print("Kurtosis: ", kurt_degarch)
print("Variance: ", std_degarch**2)