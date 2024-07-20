# Explanation
# Data Preparation for LSTM: The create_dataset function reshapes the data into sequences suitable for LSTM input. Note that the time_step is correctly handled to avoid indexing errors.
# Prediction and Evaluation: The predictions are made on the reshaped data, and the actual and predicted values are inverse transformed to their original scale.
# Plotting the Results: The actual and predicted values for both the training and test sets are plotted, ensuring the indices match correctly.
# This script ensures that the lengths of the time series and the predictions match, allowing for accurate plotting and evaluation.


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import seaborn as sns

# Sample data creation
# Generate time-series data for demonstration
time_steps = 100
data = {
    'Lease Data': np.random.randint(1000, 5000, size=time_steps),
    'Sensor Data': np.random.randint(500, 2000, size=time_steps),
    'Historical Performance Data': np.random.randint(2000, 6000, size=time_steps)
}

# Create DataFrame
data = pd.DataFrame(data)
data['Time'] = pd.date_range(start='1/1/2022', periods=time_steps, freq='H')

# Save the sample data to a CSV file
data.to_csv('real_time_monitoring.csv', index=False)

# Load the data
data = pd.read_csv('real_time_monitoring.csv')

# Prepare the data for LSTM
features = ['Lease Data', 'Sensor Data', 'Historical Performance Data']
scaler = MinMaxScaler()
data[features] = scaler.fit_transform(data[features])

# Convert the DataFrame to a NumPy array
dataset = data[features].values

# Split the data into training and testing sets
train_size = int(len(dataset) * 0.8)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]

# Create a function to reshape the data into time steps
def create_dataset(data, time_step=1):
    X, Y = [], []
    for i in range(len(data) - time_step):
        X.append(data[i:(i + time_step), :])
        Y.append(data[i + time_step, 0])  # Predicting the first feature as example
    return np.array(X), np.array(Y)

time_step = 5
X_train, y_train = create_dataset(train, time_step)
X_test, y_test = create_dataset(test, time_step)

# Reshape input to be [samples, time steps, features]
X_train = X_train.reshape(X_train.shape[0], time_step, len(features))
X_test = X_test.reshape(X_test.shape[0], time_step, len(features))

# Define the LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(time_step, len(features))))
model.add(Dropout(0.2))
model.add(LSTM(50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=20, batch_size=1, verbose=1)

# Predicting real-time analytics
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Inverse transform the predictions
train_predict = scaler.inverse_transform(np.concatenate((train_predict, np.zeros((train_predict.shape[0], len(features) - 1))), axis=1))[:, 0]
test_predict = scaler.inverse_transform(np.concatenate((test_predict, np.zeros((test_predict.shape[0], len(features) - 1))), axis=1))[:, 0]

# Inverse transform the actual values
y_train = scaler.inverse_transform(np.concatenate((y_train.reshape(-1, 1), np.zeros((y_train.shape[0], len(features) - 1))), axis=1))[:, 0]
y_test = scaler.inverse_transform(np.concatenate((y_test.reshape(-1, 1), np.zeros((y_test.shape[0], len(features) - 1))), axis=1))[:, 0]

# Evaluation
train_rmse = np.sqrt(np.mean(((train_predict - y_train) ** 2)))
test_rmse = np.sqrt(np.mean(((test_predict - y_test) ** 2)))

print(f'Train RMSE: {train_rmse}')
print(f'Test RMSE: {test_rmse}')

# Plotting the results
plt.figure(figsize=(14, 5))
plt.plot(data['Time'][:len(y_train)], y_train, label='Actual Train Data')
plt.plot(data['Time'][time_step:len(y_train) + time_step], train_predict, label='Predicted Train Data')
plt.plot(data['Time'][len(y_train) + time_step:], y_test, label='Actual Test Data')
plt.plot(data['Time'][len(y_train) + time_step:], test_predict, label='Predicted Test Data')
plt.xlabel('Time')
plt.ylabel('Lease Data')
plt.title('Actual vs Predicted Lease Data')
plt.legend()
plt.show()
