import numpy as np
from scipy.optimize import minimize

timeline=range(1,31,1)
rateline=[(6-5*np.exp(-t/30))/100 for t in timeline]

#part a
def fun_forParta(s):
    P=sum([(100*r+3)*np.exp(-1*(r+s)*t) for r,t in zip(rateline,timeline)])
    P=P+100*np.exp(-1*(rateline[-1]+s)*timeline[-1])
    return np.abs(P-82.4537464147)

res=minimize(fun=fun_forParta, x0=0.01)
print("OAS: ",res.x[0])
print("------------------------")
#part b
s=res.x[0]

def fun_forPartb(timeline,rateline,delta_r,s,delta_s):
    P=sum([(100*(r+delta_r)+3)*np.exp(-1*(r+delta_r+s+delta_s)*t) for r,t in zip(rateline,timeline)])\
      +100*np.exp(-1*(rateline[-1]+delta_r+s+delta_s)*timeline[-1])
    return P

effective_duration=(fun_forPartb(timeline,rateline,-1e-5,s,0)-fun_forPartb(timeline,rateline,1e-5,s,0))/\
                   (2e-5*fun_forPartb(timeline,rateline,0,s,0))
print("effective duration: ",effective_duration)

OASD=(fun_forPartb(timeline,rateline,0,s,-1e-5)-fun_forPartb(timeline,rateline,0,s,1e-5))/\
                   (2e-5*fun_forPartb(timeline,rateline,0,s,0))
print("OASD: ",OASD)



