import numpy as np


n=100
p=0.99
q=0.5
p100=p*100.
p_string="%0.0f"% p100

s=np.random.binomial(n,q,100000)
X=s-n*q
X.sort()
X_cal=np.insert(X,0,X[0]-1)
X_cal=np.append(X_cal,X_cal[-1]+1)

VaRp=-np.percentile(X_cal,(1-p)*100)
pv_string=p_string+"% VaR is "
print(pv_string,VaRp)

nexceed=max(np.where(X<=-VaRp)[0])
cVaRp=-np.sum([yy for yy in X if yy<=-VaRp])/(nexceed+1)
pv_string=p_string+"% cVaR is "
print(pv_string,cVaRp)

for_cvar=[]
for xx in X:
    if xx<=-VaRp:
        for_cvar.append(xx)
print(-np.mean(for_cvar))