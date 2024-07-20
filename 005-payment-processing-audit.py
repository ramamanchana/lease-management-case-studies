import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# Sample payment data
data = {
    'payment_records': [2500, 2500, 2600, 2500, 2500],
    'lease_terms': [2500, 2500, 2500, 2500, 2500],
    'anomalies': [0, 0, 1, 0, 0]
}

# Create DataFrame
df = pd.DataFrame(data)

# Random Forest Model for Anomaly Detection
X = df[['payment_records', 'lease_terms']]
y = df['anomalies']

# Train Model
model = RandomForestClassifier()
model.fit(X, y)

# Sample Input for Prediction
new_data = pd.DataFrame({
    'payment_records': [2600],
    'lease_terms': [2500]
})

# Prediction
predicted_anomaly = model.predict(new_data)
print("Predicted Anomaly:", predicted_anomaly[0])

# Plot
plt.scatter(df.index, df['anomalies'], color='blue', label='Actual')
plt.scatter(new_data.index, predicted_anomaly, color='red', label='Predicted')
plt.xlabel('Index')
plt.ylabel('Anomalies')
plt.title('Payment Processing and Auditing')
plt.legend()
plt.show()
