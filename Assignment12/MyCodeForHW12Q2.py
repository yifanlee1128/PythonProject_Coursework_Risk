import pandas as pd
import numpy as np
data=pd.read_csv("C:/Users/liyf4/Desktop/2013 Dow Jones Implieds key.csv",encoding = "ISO-8859-1")

data_of_vol=data["Implied Vol"].values[1:-2]
data_of_vol=[float(str[:-1])*0.01 for str in data_of_vol]

portfolio_vol=float(data["Implied Vol"].values[-1][:-1])*0.01

data_of_weight=data["Weight"].values[1:-2]

list_v=[i*j for i,j in zip(data_of_vol,data_of_weight)]
list_v_square=[i**2 for i in list_v]


rho_average=(portfolio_vol**2-np.sum(list_v_square))/(np.sum(list_v)**2-np.sum(list_v_square))

print(rho_average)