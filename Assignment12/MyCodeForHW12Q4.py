import pandas as pd
import numpy as np
import qrbook_funcs as qf
import scipy.stats as spst
from scipy.optimize import minimize_scalar
import scipy.optimize as scpo

ff_head = 'http://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/'
ff_foot = "_3_Factors_Daily_CSV.zip"
ff_names = ["Europe", "North_America", "Japan"]

for name_index in range(len(ff_names)):
    ffurl = ff_head + ff_names[name_index] + ff_foot
    # Skip the six header rows
    df_region = pd.read_csv(ffurl, skiprows=4)
    # Standardize name of Date column and market return column
    col0 = df_region.columns[0]
    df_region.rename(columns={col0: 'Date'}, inplace=True)
    df_region.rename(columns={"Mkt-RF": ff_names[name_index]}, inplace=True)
    # Merge into aggregate
    if name_index == 0:
        df_returns = df_region[df_region.columns[0:2]]
    else:
        df_returns = df_returns.merge(df_region[df_region.columns[0:2]],
                                      left_on='Date', right_on='Date')

# Convert to log-returns
df_logs_day = np.log(1 + df_returns[df_returns.columns[1:]] / 100)
df_logs_day.insert(0, "Date", df_returns["Date"], True)
# Remove the partial year at the end
lastyear = np.unique(list(map(int, df_logs_day["Date"].values / 10000)))[-1]
# Find the first index of the latest (partial) year in first_drop
for i in range(len(df_logs_day)):
    thisyear = int(df_logs_day.iloc[i]["Date"] / 10000)
    if thisyear == lastyear:
        first_drop = i
        break

# Remove partial year at the end
df_logs_day.drop(df_logs_day.index[first_drop:], inplace=True)

import datetime


def find_day(date_yyyymmdd):
    # Find day of week of a date
    # in the form YYYYMMDD
    date_int = int(date_yyyymmdd)
    year = int(date_int / 10000)
    month = int((date_int - year * 10000) / 100)
    day = int(date_int) % 100
    return (datetime.datetime(year, month, day).weekday())


# Done with find_day

# Convert to weekly to avoid asynchronous trading effects
# Wednesday-Wednesday
wednesday = 2  # Monday = 0
ndays = len(df_logs_day)
initialized = False
# Create empty dataframe for weekly log-returns
df_logs = df_logs_day.drop(df_logs_day.index[0:])

for t in range(ndays):
    if not initialized:
        week_sum = df_logs_day.iloc[t]
        initialized = True
    else:
        week_sum += df_logs_day.iloc[t]
    if find_day(df_logs_day.iloc[t]["Date"]) == wednesday:
        df_logs = df_logs.append(week_sum, ignore_index=True)
        df_logs.iloc[-1, 0] = int(df_logs_day.iloc[t]["Date"])
        initialized = False

periodicity = 52  # For use in later code segments

nobs = len(df_logs)

corr_matrix = df_logs[df_logs.columns[1:]].corr()
cov_matrix = df_logs[df_logs.columns[1:]].cov()
std_devs = []
for i in range(len(ff_names)):
    std_devs.append(np.sqrt(periodicity * cov_matrix.iloc[i, i]))  # annualize weekly data

zsig = np.sqrt(1 / (nobs - 3))
rsig = (np.exp(2 * zsig) - 1) / (np.exp(2 * zsig) + 1)

overallmean = np.mean(df_logs)
overallstd = np.std(df_logs)
tickerlist = df_logs.columns[1:]

gparams = []
initparams = [.12, .85, .6]
stgs = []  # Save the running garch sigma's
for it, ticker in enumerate(tickerlist):
    # Note ORDER MATTERS: make sure values are in date order
    gparams.append(qf.Garch11Fit(initparams, \
                                 df_logs.sort_values(by="Date")[ticker]))
    a, b, c = gparams[it]

    # Create time series of sigmas
    t = len(df_logs[ticker])
    minimal = 10 ** (-20)
    stdgarch = np.zeros(t)
    stdgarch[0] = overallstd[ticker]
    # Compute GARCH(1,1) stddev's from data given parameters
    for i in range(1, t):
        # Note offset - i-1 observation of data
        # is used for i estimate of std deviation
        previous = stdgarch[i - 1] ** 2
        var = c + b * previous + \
              a * (df_logs.sort_values(by="Date")[ticker][i - 1] \
                   - overallmean[ticker]) ** 2
        stdgarch[i] = np.sqrt(var)

    # Save for later de-GARCHing
    stgs.append(stdgarch)

dfeps = df_logs.sort_values(by="Date").copy()
for it, ticker in enumerate(tickerlist):
    dfeps[ticker] -= overallmean[ticker]
    for i in range(len(dfeps)):
        dfeps[ticker].iloc[i] /= stgs[it][i]
    print(ticker)
    print('    DeGARCHed Mean:', np.mean(dfeps[ticker]))

    print('    Raw annualized Std Dev:', np.sqrt(periodicity) * overallstd[ticker])
    print('    DeGARCHed Std Dev:', np.std(dfeps[ticker]))

    print('    Raw excess kurtosis:', spst.kurtosis(df_logs[ticker]))
    print('    DeGARCHed Excess Kurtosis:', spst.kurtosis(dfeps[ticker]))

InData = np.array(dfeps[tickerlist])


def MeanRevCorrObj(params):
    # Compute time series of quasi-correlation
    # matrices from InData using mean reverting
    # formula 9.45. Standardize them and apply
    # formula 9.49. Returns scalar 9.49

    # Extract parameters
    alpha, beta = params
    # Enforce bounds
    if alpha < 0 or beta < 0:
        return (10 ** 20)
    elif (alpha + beta) > .999:
        return (10 ** 20)
    obj9p39 = 0
    # Initial omega is obtained through correlation targeting
    Rlong = np.corrcoef(InData.T)
    previousq = np.identity(len(InData[0]))
    # Form new shock matrix
    for i in range(len(InData)):
        # standardize previous q matrix
        # and compute contribution to objective
        # function
        stdmtrx = np.diag([1 / np.sqrt(previousq[s, s]) \
                           for s in range(len(previousq))])
        previousr = np.matmul(stdmtrx, np.matmul(previousq, stdmtrx))
        # objective function
        obj9p39 += np.log(np.linalg.det(previousr))
        shockvec = np.array(InData[i])
        vec1 = np.matmul(shockvec, np.linalg.inv(previousr))
        # This makes obj9p39 into a 1,1 matrix
        obj9p39 += np.matmul(vec1, shockvec)

        # Update q matrix
        shockvec = np.mat(shockvec)
        shockmat = np.matmul(shockvec.T, shockvec)
        previousq = (1 - alpha - beta) * Rlong + alpha * shockmat + beta * previousq
    return (obj9p39[0, 0])


# alpha and beta positive
corr_bounds = scpo.Bounds([0, 0], [np.inf, np.inf])
# Sum of alpha and beta is less than 1
corr_linear_constraint = \
    scpo.LinearConstraint([[1, 1]], [0], [.999])

initparams = [.02, .93]

results = scpo.minimize(MeanRevCorrObj, \
                        initparams, \
                        method='trust-constr', \
                        jac='2-point', \
                        hess=scpo.SR1(), \
                        bounds=corr_bounds, \
                        constraints=corr_linear_constraint)

alpha, beta = results.x
print('Optimal alpha, beta:', alpha, beta)
print('Optimal objective function:', results.fun)
halflife = -np.log(2) / np.log(1 - alpha)
print('Half-life (years):', halflife / periodicity)


minimal=10**(-20)
xlams=[]
half_life=[]
for it in range(len(tickerlist)-1):
    tick1=tickerlist[it]
    for jt in range(it+1,len(tickerlist)):
        tick2=tickerlist[jt]
        InData=np.array(dfeps[[tick1,tick2]])
        results = scpo.minimize(MeanRevCorrObj, \
        initparams, \
        method='trust-constr', \
        jac='2-point', \
        hess=scpo.SR1(), \
        bounds=corr_bounds, \
        constraints=corr_linear_constraint)

        alpha,beta=results.x
        print('Optimal alpha, beta:',alpha,beta)
        print('Optimal objective function:',results.fun)
        halflife=-np.log(2)/np.log(1-alpha)
        print('Half-life (years):',halflife/periodicity)
        half_life.append(halflife/periodicity)

print('\nMedian half life from MacGyver Method:',np.median(half_life))