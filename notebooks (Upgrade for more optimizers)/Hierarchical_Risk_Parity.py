#!/usr/bin/env python
# coding: utf-8

# # Hierarchical Risk Parity
# 
# HRP is a modern portfolio optimization method inspired by machine learning.
# 
# The idea is that by examining the hierarchical structure of the market, we can better diversify. 
# 
# In this cookbook recipe, we will cover:
# 
# - Downloading data for HRP
# - Using HRP to find the minimum variance portfolio
# - Plotting dendrograms
# 
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pyportfolio/pyportfolioopt/blob/master/cookbook/5-Hierarchical-Risk-Parity.ipynb)
#     
# [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/PyPortfolio/PyPortfolioOpt/blob/main/cookbook/5-Hierarchical-Risk-Parity.ipynb)
#     
# [![Gradient](https://assets.paperspace.io/img/gradient-badge.svg)](https://console.paperspace.com/github/pyportfolio/pyportfolioopt/blob/master/cookbook/5-Hierarchical-Risk-Parity.ipynb)
#     
# [![Open In SageMaker Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/pyportfolio/pyportfolioopt/blob/master/cookbook/5-Hierarchical-Risk-Parity.ipynb)
# 

# ## Downloading data
# 
# HRP only requires historical returns

# In[1]:


get_ipython().system('pip install pandas numpy matplotlib yfinance PyPortfolioOpt')
import os
if not os.path.isdir('data'):
    os.system('git clone https://github.com/pyportfolio/pyportfolioopt.git')
    os.chdir('PyPortfolioOpt/cookbook')


# In[2]:


import pandas as pd
import yfinance as yf
import pypfopt

pypfopt.__version__


# In[3]:


tickers = ["BLK", "BAC", "AAPL", "TM", "WMT",
           "JD", "INTU", "MA", "UL", "CVS",
           "DIS", "AMD", "NVDA", "PBI", "TGT"]

ohlc = yf.download(tickers, period="max")
prices = ohlc["Close"]
prices.tail()


# In[4]:


from pypfopt import expected_returns

rets = expected_returns.returns_from_prices(prices)
rets.tail()


# ## HRP optimization
# 
# HRP uses a completely different backend, so it is currently not possible to pass constraints or specify an objective function.

# In[5]:


from pypfopt import HRPOpt


# In[6]:


hrp = HRPOpt(rets)
hrp.optimize()
weights = hrp.clean_weights()
weights


# In[7]:


pd.Series(weights).plot.pie(figsize=(10, 10));


# In[8]:


hrp.portfolio_performance(verbose=True);


# ## Plotting
# 
# It is very simple to plot a dendrogram (tree diagram) based on the hierarchical structure of asset returns

# In[9]:


from pypfopt import plotting

plotting.plot_dendrogram(hrp); 


# If you look at this dendogram closely, you can see that most of the clusters make a lot of sense. For example, AMD and NVDA (both semiconductor manufacturers) are grouped.

# In[ ]:




