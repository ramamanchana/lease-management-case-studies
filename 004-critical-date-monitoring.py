import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Sample lease dates
data = {
    'lease_dates': pd.date_range(start='1/1/2023', periods=12, freq='M'),
    'payment_due': [2500, 2500, 2500, 2500, 2500, 2500, 2500, 2500, 2500, 2500, 2500, 2500]
}

# Create DataFrame
df = pd.DataFrame(data)
df.set_index('lease_dates', inplace=True)

# Time-series Analysis for Date Monitoring
model = ExponentialSmoothing(df['payment_due'], trend='add', seasonal='add', seasonal_periods=12)
fit = model.fit()

# Prediction
predictions = fit.forecast(steps=3)
print("Predicted Payments for Upcoming Months:\n", predictions)

# Plot
plt.plot(df.index, df['payment_due'], label='Actual')
plt.plot(predictions.index, predictions, label='Predicted', linestyle='--')
plt.xlabel('Date')
plt.ylabel('Payment Due')
plt.title('Predictive Date Monitoring')
plt.legend()
plt.show()
