import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

#load the daily returns
daily_returns = pd.read_csv('/content/multiasset_daily_returns.csv', index_col = 0, parse_dates = True)

#define optimal weights
optimal_weights = np.array([0.10, 0.12, 0.08, 0.15, 0.10, 0.20, 0.15, 0.10])

#ensure that the dimensions match
assert len(optimal_weights) == daily_returns.shape[1], "Weight length does not match number of assets."

#calculate daily portfolio returns
portfolio_daily_returns = daily_returns @ optimal_weights
portoflio_daily_returns = pd.DataFrame(portfolio_daily_returns, index = daily_returns.index,
                                       columns = ["Optimized Portfolio"])

#calculate cumulative returns
cumulative_returns = (1 + portfolio_daily_returns).cumprod()
cumulative_returns.to_csv('/content/multiasset_cum_returns.csv')

#plot cumulative returns
plt.figure(figsize = (12, 6))
plt.plot(cumulative_returns, label = "Optimized Portfolio")
plt.title("Cumulative Returns: Optimized Portfolio")
plt.xlabel("Date")
plt.ylabel("Cumulative Returns")
plt.legend()
plt.show()

#calculate equal-weight portfolio returns
equal_weight = np.repeat(1 / daily_returns.shape[1], daily_returns.shape[1])
equal_weight_returns = daily_returns @ equal_weight
equal_weight_cum = (1 + equal_weight_returns).cumprod()
equal_weight_cum.to_csv('/content/multiasset_ewc.csv')

#plot with equal-weight returns
plt.figure(figsize = (12, 6))
plt.plot(cumulative_returns, label = "Optimized Portfolio")
plt.plot(equal_weight_cum, label = "Equal-Weight Portfolio")
plt.title("Cumulative Returns: Optimized and Equal-Weight Portfolios")
plt.xlabel("Date")
plt.ylabel("Cumulative Returns")
plt.legend()
plt.show()

#download benchmark (SPY) data
benchmark = yf.download('SPY', start = '2015-01-01', end = '2025-12-31')['Close']

#calculate the benchmark returns
benchmark_returns = np.log(benchmark / benchmark.shift(1)).dropna()
benchmark_cum = (1 + benchmark_returns).cumprod()
benchmark_cum.to_csv('/content/multiasset_benchmark.csv')

#plot the comparison
plt.figure(figsize = (12,6))
plt.plot(cumulative_returns, label = "Optimized Portfolio")
plt.plot(equal_weight_cum, label = "Equal-Weight Portfolio")
plt.plot(benchmark_cum, label = "SPY Benchmark")
plt.title("Cumulative Returns VS Benchmark")
plt.xlabel("Date")
plt.ylabel("Cumulative Returns")
plt.legend()
plt.show()

#calculate the performance metrics
def performance_metrics(returns, name = "Portfolio"):
  if isinstance(returns, pd.DataFrame):
    returns = returns.squeeze()

  ann_return = float(returns.mean() * 252)
  ann_vol = float(returns.std() * np.sqrt(252))
  sharpe_ratio = ann_return / ann_vol
  max_dd = ((1 + returns).cumprod() / (1 + returns).cumprod().cummax() - 1).min()

  print(f"{name} Performance Metrics")
  print(f"Annual Return: {ann_return:.4f}")
  print(f"Annual Volatility: {ann_vol:.4f}")
  print(f"Sharpe Ratio: {sharpe_ratio:.4f}")
  print(f"Max Drawdown: {max_dd:.4%}\n")

#optimized portfolio
performance_metrics(portfolio_daily_returns, "Optimized Portfolio")

#equal-weight portfolio
performance_metrics(equal_weight_returns, "Equal-Weight Portfolio")

#SPY benchmark portfolio
performance_metrics(benchmark_returns, "SPY Benchmark")

