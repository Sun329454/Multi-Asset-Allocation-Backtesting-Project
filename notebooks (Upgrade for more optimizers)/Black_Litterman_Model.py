#!/usr/bin/env python
# coding: utf-8

# # Black-Litterman allocation
# 
# The Black-Litterman method is a very powerful way of converting your views on asset returns, along with your uncertainty in these views, into a portfolio.
# 
# For a description of the theory, please read the [documentation page](https://pyportfolioopt.readthedocs.io/en/latest/BlackLitterman.html) and the links therein.
# 
# In this recipe, we will cover:
# 
# - Downloading data for the Black-Litterman method
# - Constructing the prior return vector based on market equilibrium
# - Two ways of constructing the uncertainty matrix
# - Combining Black-Litterman with mean-variance optimization
# 
# ## Downloading data
# 
# In addition to price data, constructing a market prior requires market-caps.
# 
# 
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pyportfolio/pyportfolioopt/blob/master/cookbook/4-Black-Litterman-Allocation.ipynb)
#     
# [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/PyPortfolio/PyPortfolioOpt/blob/main/cookbook/4-Black-Litterman-Allocation.ipynb)
#     
# [![Gradient](https://assets.paperspace.io/img/gradient-badge.svg)](https://console.paperspace.com/github/pyportfolio/pyportfolioopt/blob/master/cookbook/4-Black-Litterman-Allocation.ipynb)
#     
# [![Open In SageMaker Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/pyportfolio/pyportfolioopt/blob/master/cookbook/4-Black-Litterman-Allocation.ipynb)

# In[6]:


get_ipython().system('pip install pandas numpy matplotlib yfinance PyPortfolioOpt')
import os
if not os.path.isdir('data'):
    os.system('git clone https://github.com/pyportfolio/pyportfolioopt.git')
    os.chdir('PyPortfolioOpt/cookbook')


# In[7]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf


# In[8]:


tickers = ["MSFT", "AMZN", "NAT", "BAC", "DPZ", "DIS", "KO", "MCD", "COST", "SBUX"]


# In[9]:


ohlc = yf.download(tickers, period="max")
prices = ohlc["Close"]
prices.tail()


# In[10]:


market_prices = yf.download("SPY", period="max")["Close"]
market_prices.head()


# In[11]:


mcaps = {}
for t in tickers:
    stock = yf.Ticker(t)
    mcaps[t] = stock.info["marketCap"]
mcaps


# ## Constructing the prior

# In[12]:


import pypfopt
pypfopt.__version__


# In[13]:


from pypfopt import black_litterman, risk_models
from pypfopt import BlackLittermanModel, plotting

S = risk_models.CovarianceShrinkage(prices).ledoit_wolf()
delta = black_litterman.market_implied_risk_aversion(market_prices)
delta


# In[14]:


plotting.plot_covariance(S, plot_correlation=True);


# In[15]:


market_prior = black_litterman.market_implied_prior_returns(mcaps, delta, S)
market_prior


# In[16]:


market_prior.plot.barh(figsize=(10,5));


# ## Views
# 
# In the BL method, views are specified via the matrix P (picking matrix) and the vector Q. Q contains the magnitude of each view, while P maps the views to the assets they belong to. 
# 
# If you are providing **absolute views** (i.e a return estimate for each asset), you don't have to worry about P and Q, you can just pass your views as a dictionary.

# In[17]:


# You don't have to provide views on all the assets
viewdict = {
    "AMZN": 0.10,
    "BAC": 0.30,
    "COST": 0.05,
    "DIS": 0.05,
    "DPZ": 0.20,
    "KO": -0.05,  # I think Coca-Cola will go down 5%
    "MCD": 0.15,
    "MSFT": 0.10,
    "NAT": 0.50,  # but low confidence, which will be reflected later
    "SBUX": 0.10
}

bl = BlackLittermanModel(S, pi=market_prior, absolute_views=viewdict)


# Black-Litterman also allows for relative views, e.g you think asset A will outperform asset B by 10%. If you'd like to incorporate these, you will have to build P and Q yourself. An explanation for this is given in the [docs](https://pyportfolioopt.readthedocs.io/en/latest/BlackLitterman.html#views).

# ## View confidences
# 
# In this section, we provide two ways that you may wish to construct the uncertainty matrix. The first is known as Idzorek's method. It allows you to specify a vector/list of percentage confidences.

# In[18]:


confidences = [
    0.6,
    0.4,
    0.2,
    0.5,
    0.7, # confident in dominos
    0.7, # confident KO will do poorly
    0.7, 
    0.5,
    0.1,
    0.4
]


# In[19]:


bl = BlackLittermanModel(S, pi=market_prior, absolute_views=viewdict, omega="idzorek", view_confidences=confidences)


# In[20]:


fig, ax = plt.subplots(figsize=(7,7))
im = ax.imshow(bl.omega)

# We want to show all ticks...
ax.set_xticks(np.arange(len(bl.tickers)))
ax.set_yticks(np.arange(len(bl.tickers)))

ax.set_xticklabels(bl.tickers)
ax.set_yticklabels(bl.tickers)
plt.show()


# In[21]:


np.diag(bl.omega)


# Note how NAT, which we gave the lowest confidence, also has the highest uncertainty.
# 
# Instead of inputting confidences, we can calculate the uncertainty matrix directly by specifying 1 standard deviation confidence intervals, i.e bounds which we think will contain the true return 68% of the time. This may be easier than coming up with somewhat arbitrary percentage confidences

# In[22]:


intervals = [
    (0, 0.25),
    (0.1, 0.4),
    (-0.1, 0.15),
    (-0.05, 0.1),
    (0.15, 0.25),
    (-0.1, 0),
    (0.1, 0.2),
    (0.08, 0.12),
    (0.1, 0.9),
    (0, 0.3)
]


# In[23]:


variances = []
for lb, ub in intervals:
    sigma = (ub - lb)/2
    variances.append(sigma ** 2)

print(variances)
omega = np.diag(variances)


# ## Posterior estimates
# 
# Given the inputs, we can compute a posterior estimate of returns
# 

# In[24]:


# We are using the shortcut to automatically compute market-implied prior
bl = BlackLittermanModel(S, pi="market", market_caps=mcaps, risk_aversion=delta,
                        absolute_views=viewdict, omega=omega)


# In[25]:


# Posterior estimate of returns
ret_bl = bl.bl_returns()
ret_bl


# We can visualise how this compares to the prior and our views:

# In[26]:


rets_df = pd.DataFrame([market_prior, ret_bl, pd.Series(viewdict)], 
             index=["Prior", "Posterior", "Views"]).T
rets_df


# In[27]:


rets_df.plot.bar(figsize=(12,8));


# Notice that the posterior is often between the prior and the views. This supports the fact that the BL method is essentially a Bayesian weighted-average of the prior and views, where the weight is determined by the confidence.
# 
# A similar but less intuitive procedure can be used to produce the posterior covariance estimate:

# In[28]:


S_bl = bl.bl_cov()
plotting.plot_covariance(S_bl);


# ## Portfolio allocation
# 
# Now that we have constructed our Black-Litterman posterior estimate, we can proceed to use any of the optimizers discussed in previous recipes.

# In[29]:


from pypfopt import EfficientFrontier, objective_functions


# In[30]:


ef = EfficientFrontier(ret_bl, S_bl)
ef.add_objective(objective_functions.L2_reg)
ef.max_sharpe()
weights = ef.clean_weights()
weights


# In[31]:


pd.Series(weights).plot.pie(figsize=(10,10));


# In[32]:


from pypfopt import DiscreteAllocation

da = DiscreteAllocation(weights, prices.iloc[-1], total_portfolio_value=20000)
alloc, leftover = da.lp_portfolio()
print(f"Leftover: ${leftover:.2f}")
alloc

