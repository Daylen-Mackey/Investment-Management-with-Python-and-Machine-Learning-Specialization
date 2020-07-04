#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 13:13:49 2020

@author: dmackey
"""
# %load_ext autoreload
# %autoreload 2

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import edhec_risk_kit as erk
import scipy.stats
from scipy.stats import norm
 

#%%

hfi = erk.get_hfi_returns()

# Negative skew in the returns means you're going to get 
skew = erk.skewness(hfi).sort_values()
skew2 = scipy.stats.skew(hfi)

normal_rets = np.random.normal(0,.15, hfi.shape[0])

 
ffme = erk.get_ffme_returns()


#%% Lab Session - Semi Deviation, Var, and CVAR

# Starting with downside measures
# Negative semi-deviation would just be evaluated all the values on one
# Side of the middle

print(erk.semideviation(hfi))

print(erk.var_historic(hfi))


norm.ppf(.05) # Returns the z score

print(erk.var_gaussian(hfi))

var_list = [erk.var_gaussian(hfi),
            erk.var_gaussian(hfi, modified = True),
            erk.var_historic(hfi)
            ]
comparison = pd.concat(var_list, axis = 1)
comparison.columns = ["Gaussian", "Cornish-Fisher", "Historic"]
comparison.plot.bar()

#-- Quick Check of CVaR (Beyond VaR)

print(erk.cvar_historic(hfi))


#%% Module 1 Graded Quiz

"""
Question 1:
    What was the Annualized Return of the Lo 20 portfolio over the entire period?
    (We will use the lowest and highest quintile portfolios, 
    which are labelled ‘Lo 20’ and ‘Hi 20’ respectively.)
"""

# Reading our file of interest 
df = pd.read_csv("data/Portfolios_Formed_on_ME_monthly_EW.csv",
                  header=0, index_col=0, na_values=-99.99,
                   usecols =["Unnamed: 0", "Lo 20", "Hi 20"]
                   
                   # Alternate Below
                   # usecols =["Unnamed: 0", "Lo 10", "Hi 10"] 
                  )
df = df/100
df.index = pd.to_datetime(df.index, format="%Y%m").to_period('M')

# Calculate the returns per month first 
n_months = df.shape[0]
returns_per_month = (df+1).prod()**(1/n_months) - 1

# Calculate Annuualized Returnby compounding our monthly return

annualized_return = (returns_per_month + 1)**12 - 1

print("Lo 20 Annualized Return = " + str(annualized_return['Lo 20'] * 100) + "%")



"""
Question 2:
    What was the Annualized Volatility of the Lo 20 portfolio over the entire period? 
"""

# Already grouped in months, just find the std dev of the entire set, then 
# compound and square root
annualized_vol = df.std()*(12**0.5)

print("Lo 20 Annualized Volatility = " + str(annualized_vol['Lo 20'] * 100) + "%")


"""
Question 3:
    What was the Annualized Return of the Hi 20 portfolio over the entire period?
"""

print("Hi 20 Annualized Return = " + str(annualized_return['Hi 20'] * 100) + "%")


"""
Question 4:
    What was the Annualized Volatility of the Hi 20 portfolio over the entire period ?
"""

print("Hi 20 Annualized Volatility = " + str(annualized_vol['Hi 20'] * 100) + "%")


"""
Question 5:
    What was the Annualized Return of the Lo 20 portfolio over the 
    period 1999 - 2015 (both inclusive)?
"""

# Defining range 1 as 1999 -  2015

range1 = slice("1999", "2015")

# Creating dataframe for our new range
df_range1 = df[range1]

n_months_range1 = df_range1.shape[0]
returns_per_month_range1 = (df_range1+1).prod()**(1/n_months_range1) - 1

# Calculate Annuualized Returnby compounding our monthly return

annualized_return_range1 = (returns_per_month_range1 + 1)**12 - 1

print("Range 1 Lo 20 Annualized Return = " + 
      str(annualized_return_range1['Lo 20'] * 100) + "%")

"""
Question 6:
    What was the Annualized Volatility of the Lo 20 portfolio over 
    the period 1999 - 2015 (both inclusive)? 
"""
annualized_vol_range1 = df_range1.std()*(12**0.5)

print("Range 1 Lo 20 Annualized Volatility = " + 
      str(annualized_vol_range1['Lo 20'] * 100) + "%")

"""
Question 7:
    What was the Annualized Return of the Hi 20 portfolio over the 
    period 1999 - 2015 (both inclusive)?
"""

print("Range 1 Hi 20 Annualized Return = " + 
      str(annualized_return_range1['Hi 20'] * 100) + "%")

"""
Question 8:
    What was the Annualized Volatility of the Hi 20 portfolio over the 
    period 1999 - 2015 (both inclusive)? 
"""
print("Range 1 Hi 20 Annualized Volatility = " + 
      str(annualized_vol_range1['Hi 20'] * 100) + "%")


"""
Question 9:
    What was the Max Drawdown (expressed as a positive number) experienced over the 1
    999-2015 period in the SmallCap (Lo 20) portfolio?
"""
# Note that Max Drawdown will essentially be a minimum!

# First, we need to create a wealth index

# Function taken from the class lab section
def drawdown(return_series: pd.Series):
    """Takes a time series of asset returns.
       returns a DataFrame with columns for
       the wealth index, 
       the previous peaks, and 
       the percentage drawdown
    """
    wealth_index = 1000*(1+return_series).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdowns = (wealth_index - previous_peaks)/previous_peaks
    return pd.DataFrame({"Wealth": wealth_index, 
                         "Previous Peak": previous_peaks, 
                         "Drawdown": drawdowns})

drawdown_Lo20 = drawdown(df_range1["Lo 20"])

print("Range 1 Lo 20 Max Drawdown = " + 
      str(drawdown_Lo20.Drawdown.min() * -100) + "%")
# Uncomment below to plot for visual confirmation

# drawdown_Lo20.plot()
# drawdown_Lo20.Drawdown.plot()


"""
Question 10:
    At the end of which month over the period 1999-2015 did 
    that maximum drawdown on the SmallCap (Lo 20) portfolio occur?
"""

print("Range 1 Lo 20 Max Drawdown Month = " + 
      str(drawdown_Lo20.Drawdown.idxmin()))

"""
Question 11:
    What was the Max Drawdown (expressed as a positive number) experienced over the 1
    999-2015 period in the SmallCap (Lo 20) portfolio?
"""


drawdown_Hi20 = drawdown(df_range1["Hi 20"])

print("Range 1 Hi 20 Max Drawdown = " + 
      str(drawdown_Hi20.Drawdown.min() * -100) + "%")


"""
Question 12:
    Over the period 1999-2015, at the end of which month 
    did that maximum drawdown of the LargeCap (Hi 20) portfolio occur?
"""


print("Range 1 Hi 20 Max Drawdown Month = " + 
      str(drawdown_Hi20.Drawdown.idxmin()))


"""
Question 13:
    For the remaining questions, use the EDHEC Hedge Fund Indices data set 
    that we used in the lab assignment and load them into Python.
    Looking at the data since 2009 (including all of 2009) through 2018 
    which Hedge Fund Index has exhibited the highest semideviation?
"""
print("\nNew data set: \n ")

hfi = erk.get_hfi_returns()

range2 = slice("2009","2018")
hfi_range2 = hfi[range2]

semi_dev = erk.semideviation(hfi_range2)
print("Highest semideviation: " + str(semi_dev.idxmax()))

"""
Question 14:
    Looking at the data since 2009 (including all of 2009) which Hedge Fund 
    Index has exhibited the lowest semideviation?
"""
print("Highest semideviation: " + str(semi_dev.idxmin()))

"""
Question 15:
    Looking at the data since 2009 (including all of 2009) which 
    Hedge Fund Index has been most negatively skewed?
    
    Apparently I got this question wrong :(
"""

skew_frame = erk.skewness(hfi)

print("Highest negative skewness: " + str(skew_frame.idxmin()))

"""
Question 16:
    Looking at the data since 2000 (including all of 2000) through 2018 
    which Hedge Fund Index has exhibited the highest kurtosis?
"""
kurt_frame = erk.kurtosis(hfi)

print("Highest Kurtosis: " + str(kurt_frame.idxmax()))



