#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 13:54:34 2020

@author: dmackey
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import edhec_risk_kit_129 as erk
import scipy.stats
from scipy.stats import norm

# %load_ext autoreload
# %autoreload 2
# %matplotlib inline
#%%


ind = erk.get_ind_returns()

drawdown = erk.drawdown(ind['Food'])

# drawdown.Drawdown.plot()

erk.sharpe_ratio(ind, .03, 12).plot.bar()
#%%

ind = erk.get_ind_returns()

date_range = slice("1996", "2000")
er = erk.annualize_rets(ind[date_range],12) #expected returns vector
cov = ind[date_range].cov() #covariance matrix

cols = ["Food", "Beer", "Smoke", "Coal"]
#Used to calculate the return of the portfolio with equal weighting
weights = np.repeat(1/4,4)
even_return = erk.portfolio_return(weights,er[cols])

even_vol = erk.portfolio_vol(weights, cov.loc[cols,cols])


#%% 2 Asset Frontier Construction

cols = ["Games", "Fin"]
n_points = 50

weights = [np.array([w, 1-w]) for w in np.linspace(0,1,n_points)
    ]

rets = [erk.portfolio_return(w, er[cols]) for w in weights]
vols = [erk.portfolio_vol(w, cov.loc[cols,cols]) for w in weights]

ef = pd.DataFrame({
    "R" : rets,
    "Vol" : vols
    })

ef.plot.scatter(x = "Vol",y = "R")


#%% N-Asset Efficient Frontier

ind = erk.get_ind_returns()
date_range = slice("1996", "2000")
er = erk.annualize_rets(ind[date_range],12) #expected returns vector
cov = ind[date_range].cov() #covariance matrix

# erk.plot_ef2(20, er[cols], cov.loc[cols,cols])

l = ["Smoke", "Fin", "Games", "Coal"]
erk.plot_ef(25, er[l], cov.loc[l,l])


#%% 
"""
Finding the Max Sharpe Ratio Portfolio!!
"""
ind = erk.get_ind_returns()
date_range = slice("1996", "2000")
er = erk.annualize_rets(ind[date_range],12) #expected returns vector
cov = ind[date_range].cov() #covariance matrix

erk.plot_ef(20, er, cov,show_cml = True,show_ew = True, show_gmv = True,
            riskfree_rate = -.5
            )



#%% MODULE 2 GRADED QUIZ!
"""
Question 1:
Use the EDHEC Hedge Fund Indices data set that we used in the lab assignment 
as well as in the previous week’s assignments. Load them into Python and 
perform the following analysis based on data since 2000 (including all of 
2000): What was the Monthly Parametric Gaussian VaR at the 1% level (as a 
+ve number) of the Distressed Securities strategy?
"""







hfi = erk.get_hfi_returns()
hfi_2000_dis = hfi.loc["2000":, "Distressed Securities"]

q1ans = erk.var_gaussian(hfi_2000_dis,level = 1)

print("Q1Ans = " + str(q1ans*100) + "%")

"""
Question 2:
Use the same data set at the previous question. What was the 1% VaR for 
the same strategy after applying the Cornish-Fisher Adjustment?
"""

q2ans = erk.var_gaussian(hfi_2000_dis,level = 1,modified = True)

print("Q2Ans = " + str(q2ans*100) + "%")

"""
Question 3:
Use the same dataset as the previous question. 
What was the Monthly Historic VaR at the 1% level (as a +ve number) of t
he Distressed Securities strategy?
"""

q3ans = erk.var_historic(hfi_2000_dis,level = 1)

print("Q3Ans = " + str(q3ans*100) + "%")



"""
Question 4:
Next, load the 30 industry return data using the erk.get_ind_returns() 
function that we developed during the lab sessions. For purposes of the 
remaining questions, use data during the 5 year period 2013-2017 
(both inclusive) to estimate the expected returns as well as the covariance 
matrix. To be able to respond to the questions, you will need to build the
 MSR, EW and GMV portfolios consisting of the “Books”, “Steel”, "Oil", and 
 "Mines" industries. Assume the risk free rate over the 5 year period is 10%.
What is the weight of Steel in the EW Portfolio?
"""
ind = erk.get_ind_returns()
df = ind["2013" : "2017"]
cols = ["Books", "Steel", "Oil", "Mines"]

# Weight in an EW portfolio are *EQUAL*

q4ans = 1 / len(cols)
print("Q4Ans = " + str(q4ans*100) + "%")

"""
Question 5:
What is the weight of the largest component of the MSR portfolio?
"""
cov = df.cov()
er = erk.annualize_rets(df, 12)

msr_weights = np.around(erk.msr(.1, er[cols], cov.loc[cols,cols]))
q5ans = msr_weights.max()
print("Q5Ans = " + str(q5ans*100) + "%")



"""
Question 6:
Which of the 4 components has the largest weight in the MSR portfolio?
"""
q6ans = cols[msr_weights.argmax()]

print("Q6Ans = " + str(q6ans))

"""
Question 7:
How many of the components of the MSR portfolio have non-zero weights?
"""
q7ans = np.count_nonzero(msr_weights)
print("Q7Ans = " + str(q7ans))

"""
Question 8:
What is the weight of the largest component of the GMV portfolio?
"""
gmv_weights = np.around(erk.gmv(cov.loc[cols,cols]))
q8ans = gmv_weights.max()

print("Q8Ans = " + str(q8ans*100) + "%")

"""
Question 9:
Which of the 4 components has the largest weight in the GMV portfolio?
"""

q9ans = cols[gmv_weights.argmax()]
print("98Ans = " + str(q9ans))


"""
Question 10:
 How many of the components of the GMV portfolio have non-zero weights?
"""

q10ans = np.count_nonzero(gmv_weights)
print("Q9Ans = " + str(q10ans))




