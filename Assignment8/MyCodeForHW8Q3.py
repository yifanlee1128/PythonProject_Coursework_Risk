import pandas as pd
import qrbook_funcs as qf
import numpy as np
import scipy.stats as spst


def BoxM(T1, T2, s1, s2):
    # Tests for equality of two covariance matrices, s1 and s2
    # T1 and T2 are numbers of observations for s1 and s2
    # Returns M statistic and p-value

    # Make sure dimension is common
    if len(s1) != len(s2):
        print("Error: different dimensions in Box M Test:", len(s1), len(s2))
        return (0, 0)

    # Matrices are pxp
    p = len(s1)

    # Form the combined matrix
    scomb = (T1 * s1 + T2 * s2) / (T1 + T2)

    # Box M statistic
    Mstat = (T1 + T2 - 2) * np.log(np.linalg.det(scomb)) - (T1 - 1) * np.log(np.linalg.det(s1)) - (T2 - 1) * np.log(
        np.linalg.det(s2))

    # Multipliers from equation (49) in Box 1949.
    A1 = (2 * p ** 2 + 3 * p - 1) / (6 * (p + 1))
    A1 *= (1 / (T1 - 1) + 1 / (T2 - 1) - 1 / (T1 + T2 - 2))

    A2 = (p - 1) * (p + 2) / 6
    A2 *= (1 / (T1 - 1) ** 2 + 1 / (T2 - 1) ** 2 - 1 / (T1 + T2 - 2) ** 2)

    discrim = A2 - A1 ** 2

    # Degrees of freedom
    df1 = p * (p + 1) / 2

    if discrim <= 0:
        # Use chi-square (Box 1949 top p. 329)
        test_value = Mstat * (1 - A1)
        p_value = 1 - spst.chi2.cdf(test_value, df1)
    else:
        # Use F Test (Box 1949 equation (68))
        df2 = (df1 + 2) / discrim
        b = df1 / (1 - A1 - (df1 / df2))
        test_value = Mstat / b
        p_value = 1 - spst.f.cdf(test_value, df1, df2)

    return (Mstat,test_value, p_value)


def levene(T1,T2,x1,x2):

    m1=np.average(x1)
    m2=np.average(x2)
    z1j=[np.abs(x1[j]-m1) for j in range(T1)]
    z2j=[np.abs(x2[j]-m2) for j in range(T2)]
    z1=np.average(z1j)
    z2=np.average(z2j)

    levene_mult=(T1+T2-2)*T1*T2/(T1+T2)

    levene_denom=np.sum((z1j-z1)**2)+np.sum((z2j-z2)**2)
    levene_stat=levene_mult*(z1-z2)**2/levene_denom

    p_value = 1 - spst.f.cdf(levene_stat, 1, T1+T2-2)

    return(levene_stat,p_value)


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

#Mean vector and covariance matrix are inputs to efficient frontier calculations
d=np.array(difflgs)
m=np.mean(d,axis=0)
T1=len(d)
c=np.cov(d.T)
S1=c

w = np.array([1 / 3] * 3).T
# apply transform to get returns at portfolio level
portfolio = np.log(1 + np.dot(np.exp(difflgs) - 1, w))


statnames, metrics, table = qf.StatsTable(np.exp(portfolio) - 1)

count=metrics[0]
chol=np.linalg.cholesky(c)
seed=np.random.seed(12345678)
s_trial=np.random.normal(0,1,size=[int(count),3])
logr_trial=np.matmul(chol,s_trial.T).T+m

#logr_trial has Monte Carlo log-returns; transform to returns
r_trial=np.exp(logr_trial)-1

T2=len(r_trial)
S2=np.cov(r_trial.T)

M12,test_value,p_value=BoxM(T1,T2,S1,S2)
print(M12,test_value,p_value)

levene_stat,p_value=levene(T1,T2,d,r_trial)
print(levene_stat,p_value)

a0,b0=levene(T1,T2,d[:,0],r_trial[:,0])
a1,b1=levene(T1,T2,d[:,1],r_trial[:,1])
a2,b2=levene(T1,T2,d[:,2],r_trial[:,2])
print(a0,b0)
print(a1,b1)
print(a2,b2)
print(T1,T2)