import numpy as np
from scipy import stats


u_list=[stats.norm.ppf(1-np.exp(-0.07*2)),stats.norm.ppf(1-np.exp(-0.1*2))]
cov_matrix=[[1,0.2],[0.2,1]]
res=stats.multivariate_normal.cdf(u_list,[0,0],cov_matrix)
print(res)