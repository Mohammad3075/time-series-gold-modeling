import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from arch import arch_model

path = r"D:\Coding\gold price simulaiton.csv"
df = pd.read_csv(path)

# 2. Clean the Price column (Remove commas and convert to numbers)
df['Price'] = df['Price'].astype(str).str.replace(',', '')
df['Price'] = pd.to_numeric(df['Price'], errors='coerce')

# 3. Sort by Date to ensure the time series is in order
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')

# 4. Calculate Log Returns (Multiplied by 100 for better scaling)
# Log returns are standard for GARCH to ensure the data is stationary.
df['Return'] = 100 * (np.log(df['Price']) - np.log(df['Price'].shift(1)))
df = df.dropna(subset=['Return'])

# 5. Fit the ARIMA(1,0,1)-GARCH(1,1) Model
# mean='AR' (for the AR component), lags=1 (for the AR(1) component)
# vol='Garch' (for the volatility component), p=1, q=1 (standard GARCH)
model = arch_model(df['Return'], mean='AR', lags=1, vol='Garch', p=1, q=1)
results = model.fit(disp='on')

# 6. Display the results
print(results.summary())

# 7. Visualize the Conditional Volatility
# This shows how "risk" or volatility changes over time for Gold.
results.plot()
plt.show()
# 1. Forecast the next 252 trading days (1 year)
horizon = 252
forecasts = results.forecast(horizon=horizon, reindex=False)

# 2. Extract the predicted Mean and Variance
forecast_mean = forecasts.mean.iloc[-1]
forecast_variance = forecasts.variance.iloc[-1]

# 3. Calculate the Confidence Intervals (95%)
# 1.96 is the Z-score for a 95% confidence level
std_dev = np.sqrt(forecast_variance)
upper_bound = forecast_mean + (1.96 * std_dev)
lower_bound = forecast_mean - (1.96 * std_dev)

# 4. Plotting the Volatility Forecast
plt.figure(figsize=(10, 6))
plt.plot(range(1, horizon + 1), std_dev, color='red',
         label='Forecasted Volatility (Risk)')
plt.title('Gold Volatility Forecast (Next 1 Year)')
plt.xlabel('Days into the Future')
plt.ylabel('Expected Standard Deviation')
plt.legend()
plt.show()

print(f"Projected daily return: {forecast_mean.mean():.4f}%")
print(f"Average projected daily risk: {std_dev.mean():.4f}%")

# 1. Define horizons (Trading days: ~21 per month)
h_3m = 63
h_1y = 252

# 2. Generate the forecast from your 'results' object
forecasts = results.forecast(horizon=h_1y, reindex=False)

# 3. Get the predicted mean returns and the volatility (standard deviation)
# The mean is daily return; the variance is daily risk squared
mean_forecast = forecasts.mean.values[-1]
vol_forecast = np.sqrt(forecasts.variance.values[-1])

# 4. Calculate Price Levels (Assuming last known price is from your data)
last_price = df['Price'].iloc[-1]

# Project the price using the cumulative sum of log returns
# We use the mean return and the forecasted volatility
time_steps = np.arange(1, h_1y + 1)
price_path = last_price * np.exp(np.cumsum(mean_forecast / 100))

# Upper and Lower 95% Confidence Intervals (The "Danger Zone")
# 1.96 standard deviations covers 95% of probable outcomes
# Change 1.96 to 1.28 for an 80% Confidence Interval
upper_band_80 = last_price * \
    np.exp(np.cumsum(mean_forecast / 100) + 1.28 *
           np.sqrt(np.cumsum(vol_forecast**2 / 10000)))
lower_band_80 = last_price * \
    np.exp(np.cumsum(mean_forecast / 100) - 1.28 *
           np.sqrt(np.cumsum(vol_forecast**2 / 10000)))

# 5. Plotting
plt.figure(figsize=(12, 6))
plt.plot(time_steps, price_path, label='Forecasted Price (Mean)', color='blue')
plt.fill_between(time_steps, lower_band_80, upper_band_80,
                 color='blue', alpha=0.1, label='95% Confidence Interval')
plt.title('Gold Price Forecast: Next 1 Year')
plt.xlabel('Days into Future')
plt.ylabel('Price')
plt.axvline(x=63, color='red', linestyle='--', label='3-Month Mark')
plt.legend()
plt.show()
