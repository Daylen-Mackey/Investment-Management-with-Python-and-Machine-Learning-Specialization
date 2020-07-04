import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import edhec_risk_kit as erk
import scipy.stats
from scipy.stats import norm

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

def get_ffme_returns():
    """
    Load the Fama-French Dataset for the returns of the Top and Bottom Deciles by MarketCap
    """
    me_m = pd.read_csv("data/Portfolios_Formed_on_ME_monthly_EW.csv",
                       header=0, index_col=0, na_values=-99.99)
    rets = me_m[['Lo 10', 'Hi 10']]
    rets.columns = ['SmallCap', 'LargeCap']
    rets = rets/100
    rets.index = pd.to_datetime(rets.index, format="%Y%m").to_period('M')
    return rets



def get_hfi_returns():
    """
    Load the Fama-French Dataset for the returns of the Top and Bottom Deciles by MarketCap
    """
    hfi = pd.read_csv("data/edhec-hedgefundindices.csv",
                       header=0, index_col=0, na_values=-99.99,parse_dates = True)
    # rets = me_m[['Lo 10', 'Hi 10']]
    # rets.columns = ['SmallCap', 'LargeCap']
    
    hfi = hfi/100
    hfi.index = hfi.index.to_period('M')
    return hfi

def skewness(r):
    demeaned_r = r - r.mean()
    # use the population standard deviation, so set dof=0
    sigma_r = r.std(ddof=0)
    exp = (demeaned_r**3).mean()
    return exp/sigma_r**3

def kurtosis(r):
    demeaned_r = r-r.mean()
    sigma_r = r.std(ddof = 0)
    #ddof = 0 uses the population standard deviation instead of sample
    exp = (demeaned_r**4).mean()
    
    return exp/sigma_r**4
message = "Just Testing"

def is_normal(r, level = .01):
    statistic, p_value = scipy.stats.jarque_bera(r)
    
    return p_value > level

def semideviation(r):
    
    is_negative = r < 0
    return r[is_negative].std(ddof = 0)

def var_historic(r, level = 5):
    
    if isinstance(r, pd.DataFrame):
        return r.aggregate(var_historic, level = level)
    elif isinstance(r, pd.Series):
        return -np.percentile(r,level)
    else:
        raise TypeError("Expected r to be Series or DataFrame")
        
def var_gaussian (r, level = 5, modified = False):
    z = norm.ppf(level/100)
    
    if modified:
        # modify the Z score based on observed skewness and kurtosis
        
        s = skewness(r)
        k = kurtosis(r)
        z = (z + (z**2 - 1) * s/6 + 
             (z**3 - 3*z) * (k-3) / 24 - 
             (2*z**3 - 5*z) * (s**2)/36
             )
        
             
    return -(r.mean() + z*r.std(ddof = 0))


def cvar_historic(r, level = 5):
    
    if isinstance(r, pd.DataFrame):
        is_beyond = r <= -var_historic(r, level = level)
        return -r[is_beyond].mean()
    elif isinstance(r, pd.Series):
        return r.aggregate(cvar_historic,level = level)
    else:
        raise TypeError("Expected r to be Series or DataFrame")
        