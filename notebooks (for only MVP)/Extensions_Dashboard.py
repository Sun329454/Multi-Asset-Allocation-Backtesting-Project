import cvxpy as cp
import pandas as pd
import numpy as np

#load the daily returns
daily_returns = pd.read_csv('/content/multiasset_daily_returns.csv')

#print robust optimal weights
print("Robust Optimal Weights: ")
for i in range(n):
  print(f"{daily_returns.columns[i]}: {w.value[i]:.4f}")

#assume previous portfolio weights
prev_weights = np.array([0.10, 0.12, 0.08, 0.15, 0.10, 0.20, 0.15, 0.10])

#define turnover limit
turnover_limit = 0.10 #set at 10%

#re-run optimization with turnover constraint
w = cp.Variable(n)

objective = cp.Minimize(cp.quad_form(w, sigma))
constraints = [cp.sum(w) == 1, w >= 0, w <= 0.20,
               cp.norm(w - prev_weights, 1) <= turnover_limit, mu @ w >= ret_target]

problem = cp.Problem(objective, constraints)
problem.solve()

#print optimal weights with turnover constraint
print("Optimal Weights with Turnover Constraint: ")
for i in range(n):
  print(f"{daily_returns.columns[i]}: {w.value[i]:.4f}")

