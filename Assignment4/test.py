import numpy as np
from scipy.stats import spearmanr,pearsonr
a = [0,10,-0.1]

b = [-5,2,3]

print(spearmanr(a,b))
print(pearsonr(a,b))
