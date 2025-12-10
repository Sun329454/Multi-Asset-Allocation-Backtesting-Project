import cvxpy as cp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#load daily returns
daily_returns = pd.read_csv('/content/multiasset_daily_returns.csv', index_col = 0, parse_dates = True)

#compute annualized mean returns
mean_daily_returns = daily_returns.mean()
annual_returns = mean_daily_returns * 252 #assumption: 252 trading days

#calculate covariance
cov_matrix = daily_returns.cov()
annual_cov_matrix = cov_matrix * 252 #assumption: 252 trading days

#convert to NumPy arrays for optimization
mu = annual_returns.values
sigma = annual_cov_matrix.values

#I have set the following constraints:
#sum of weights = 1
#no short selling (weights >= 0)
#max allocation per asset (<= 20%)

#number of assets
n = len(mu)

#define optimization variables
w = cp.Variable(n)

#define target return
ret_target = 0.05 #5% annual return

#define objective -> minimize portfolio variance
portfolio_variance = cp.quad_form(w, sigma)
objective = cp.Minimize(portfolio_variance)

#define given constraints
constraints = [cp.sum(w) == 1, w >= 0, w <= 0.20, mu @ w >= ret_target]

#formulate problem
problem = cp.Problem(objective, constraints)

#solve the problem
problem.solve()
print("Problem status:", problem.status)

#proceed only if solution exists
if w.value is not None:
    optimal_weights = w.value

    portfolio_return = np.dot(mu, optimal_weights)
    portfolio_volatility = np.sqrt(np.dot(optimal_weights.T, np.dot(sigma, optimal_weights)))

    print(f"Expected Portfolio Return: {portfolio_return:.4f}")
    print(f"Expected Portfolio Volatility: {portfolio_volatility:.4f}")
else:
    print("Optimization failed. Adjust target return or constraints.")

#plot portfolio weights
plt.figure(figsize = (10, 6))
plt.bar(daily_returns.columns, optimal_weights)
plt.title("Optimal Portfolio Weights")
plt.ylabel("Weight")
plt.xticks(rotation = 45)
plt.show()

#generate efficient frontier
target_returns = np.linspace(0.02, 0.12, 20)

#lists to store results
portfolio_vols = []
portfolio_returns = []

for r in target_returns:
  W = cp.Variable(n)
  objective = cp.Minimize(cp.quad_form(W, sigma))
  constraints = [cp.sum(W) == 1, W >= 0, W <= 0.20, mu @ W >= r]
  prob = cp.Problem(objective, constraints)
  prob.solve()

  if W.value is not None:
    portfolio_vols.append(np.sqrt(np.dot(W.value.T, np.dot(sigma, W.value))))
    portfolio_returns.append(np.dot(mu, W.value))
  else:
    portfolio_vols.append(None)
    portfolio_returns.append(None)

#convert to arrays and filter None values
portfolio_vols = [np.nan if v is None else v for v in portfolio_vols]
portfolio_returns = [np.nan if r is None else r for r in portfolio_returns]

portfolio_vols = np.array(portfolio_vols)
portfolio_returns = np.array(portfolio_returns)

valid = ~np.isnan(portfolio_vols)

#plot efficent frontier
plt.figure(figsize = (10, 6))
plt.plot(portfolio_vols[valid], portfolio_returns[valid], 'o-')
plt.xlabel("Portfolio Volatility")
plt.ylabel("Portfoilo Returns")
plt.title("Efficient Frontier")
plt.show()

