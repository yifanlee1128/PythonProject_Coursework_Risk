import numpy as np
from scipy.optimize import minimize

timeline=range(1,31,1)
rateline=[(6-5*np.exp(-t/30))/100 for t in timeline]

qd=0.02
coupon=2
R=50

#Parta
P=sum([R*qd/(1-qd)*(1-qd)**i/(1+r)**i+coupon*(1-qd)**i/(1+r)**i for i,r in zip(timeline,rateline)])\
  +100*(1-qd)**timeline[-1]/(1+rateline[-1])**timeline[-1]
print("Pdef: ",P)
print("------------------------")

#Partb
def fun_forPartb(s):
    P1=sum([coupon/(1+r+s)**i for r,i in zip(rateline,timeline)])
    P1=P1+100/(1+rateline[-1]+s)**timeline[-1]
    return np.abs(P1-P)

res=minimize(fun=fun_forPartb, x0=0.01)
print("spread: ",res.x[0])