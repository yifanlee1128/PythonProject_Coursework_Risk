import numpy as np
from scipy.optimize import minimize

t=np.linspace(0,10,11)
coupon=2
R=40

rateline=np.interp(t,[0,10],[0.01,0.06])[1:]
timeline=t[1:]

def_free_bond_price=sum([coupon/(1+r)**i for i,r in zip(timeline,rateline)])+100/(1+rateline[-1])**timeline[-1]

print("the price of default-free bnond is: ",def_free_bond_price)

def func(qd):
    P1 = sum([R * qd / (1 - qd) * (1 - qd) ** i / (1 + r) ** i + coupon * (1 - qd) ** i / (1 + r) ** i for i, r in
             zip(timeline, rateline)]) \
        + 100 * (1 - qd) ** timeline[-1] / (1 + rateline[-1]) ** timeline[-1]
    return np.abs(P-P1)

percent=[0.02,0.05,0.1]

for p in percent:
    P=(1-p)*def_free_bond_price
    res=minimize(func,0.01)
    print("when the price of the bond is ",p," less than a default-free bond,the risk-neutral prob of default is: ",res.x[0])


