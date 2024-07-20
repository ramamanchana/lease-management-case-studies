import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Sample lease data for reporting
data = {
    'lease_id': [1, 2, 3, 4, 5],
    'monthly_rent': [2500, 2600, 2700, 2500, 2550],
    'compliance_status': [1, 1, 0, 1, 1]
}

# Create DataFrame
df = pd.DataFrame(data)

# Linear Regression Model for Compliance Reporting
X = df[['monthly_rent']]
y = df['compliance_status']

# Train Model
model = LinearRegression()
model.fit(X, y)

# Sample Input for Prediction
new_data = pd.DataFrame({
    'monthly_rent': [2650]
})

# Prediction
predicted_compliance = model.predict(new_data)
print("Predicted Compliance Status:", predicted_compliance[0])

# Plot
plt.scatter(df['monthly_rent'], df['compliance_status'], color='blue', label='Actual')
plt.scatter(new_data['monthly_rent'], predicted_compliance, color='red', label='Predicted')
plt.xlabel('Monthly Rent')
plt.ylabel('Compliance Status')
plt.title('Compliance Reporting')
plt.legend()
plt.show()
