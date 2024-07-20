# Explanation
# Sample Data Creation: The script creates a sample dataset containing lease dates, notification preferences, historical compliance data, and hypothetical future critical dates. This dataset is saved as lease_dates.csv.
# Data Loading and Preprocessing: The script reads the sample data, preprocesses the dates by converting them to ordinal format, and encodes categorical data.
# Feature Engineering and Scaling: The script scales the data using MinMaxScaler and splits it into training and testing sets.
# Model Definition and Training: An LSTM model is defined and trained to predict upcoming critical lease dates.
# Model Evaluation: The model's performance is evaluated using Root Mean Squared Error (RMSE), and the predictions are compared with actual dates.
# Visualization: The results are visualized using line plots and scatter plots to show the distribution of actual vs predicted dates.

# Before running the script, ensure you have the necessary libraries installed:

# pip install pandas numpy matplotlib seaborn scikit-learn keras tensorflow



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import seaborn as sns

# Sample data creation
data = {
    'Lease Dates': pd.date_range(start='1/1/2021', periods=100, freq='M'),
    'Notification Preferences': np.random.choice(['Email', 'SMS', 'None'], size=100),
    'Historical Compliance Data': np.random.randint(0, 2, size=100)
}

# Create DataFrame
data = pd.DataFrame(data)
data['Upcoming Critical Dates'] = data['Lease Dates'] + pd.DateOffset(months=6)  # Hypothetical future critical dates

# Save the sample data to a CSV file
data.to_csv('lease_dates.csv', index=False)

# Load the data
data = pd.read_csv('lease_dates.csv', parse_dates=['Lease Dates', 'Upcoming Critical Dates'])

# Feature Engineering
data['Lease Dates'] = data['Lease Dates'].map(pd.Timestamp.toordinal)
data['Upcoming Critical Dates'] = data['Upcoming Critical Dates'].map(pd.Timestamp.toordinal)

# Preparing Data for Time Series Prediction
data = data.sort_values('Lease Dates')
X = data[['Lease Dates', 'Notification Preferences', 'Historical Compliance Data']].values
y = data['Upcoming Critical Dates'].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
X[:, 1] = label_encoder.fit_transform(X[:, 1])

# Scale data
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
y_scaled = scaler.fit_transform(y.reshape(-1, 1))

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# Reshape data for LSTM [samples, time steps, features]
X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

# Define the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=1)

# Predicting upcoming critical dates
y_pred = model.predict(X_test)
y_pred_rescaled = scaler.inverse_transform(y_pred)
y_test_rescaled = scaler.inverse_transform(y_test)

# Evaluation
rmse = np.sqrt(mean_squared_error(y_test_rescaled, y_pred_rescaled))
print(f'Root Mean Squared Error: {rmse}')

# Visualization
plt.figure(figsize=(14, 5))
plt.plot(y_test_rescaled, color='blue', label='Actual Upcoming Critical Dates')
plt.plot(y_pred_rescaled, color='red', label='Predicted Upcoming Critical Dates')
plt.title('Actual vs Predicted Upcoming Critical Dates')
plt.xlabel('Sample Index')
plt.ylabel('Dates (ordinal format)')
plt.legend()
plt.show()

# Scatter plot to show the distribution of actual vs predicted dates
sns.scatterplot(x=y_test_rescaled.flatten(), y=y_pred_rescaled.flatten())
plt.xlabel('Actual Upcoming Critical Dates')
plt.ylabel('Predicted Upcoming Critical Dates')
plt.title('Scatter plot of Actual vs Predicted Dates')
plt.show()
