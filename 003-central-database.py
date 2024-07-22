# Case Study: Centralized Database Management in Real Estate Lease Abstraction

"""
Description:
Storing abstracted lease information in a centralized, secure database for easy access and management. This ensures that all lease data is accessible from a single platform, improving data accuracy and facilitating efficient lease management.

Model: Database Management Systems with ML-enhanced data retrieval

Data Input: Extracted Lease Data, Metadata

Prediction: Data Retrieval, Data Accuracy

Recommended Model: Database Management Systems integrated with ML models for enhanced data accuracy and retrieval

Customer Value Benefits: Operational Efficiency, Strategic Decision Support

Use Case Implementation:
By centralizing lease information in a secure database, the system improves data accuracy and accessibility. Integrating ML models enhances data retrieval, ensuring efficient management and strategic decision support.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
import seaborn as sns

# Sample data creation
np.random.seed(42)
extracted_lease_data = np.random.rand(100, 5)  # Dummy data for lease features
metadata = np.random.rand(100, 5)  # Dummy data for metadata features
data_accuracy = np.random.rand(100)  # Dummy target variable for data accuracy

# Create DataFrame
data = pd.DataFrame(np.hstack((extracted_lease_data, metadata)), columns=[f'Feature_{i}' for i in range(10)])
data['Data Accuracy'] = data_accuracy

# Save the sample data to a CSV file
data.to_csv('centralized_database_management.csv', index=False)

# Load the data
data = pd.read_csv('centralized_database_management.csv')

# Prepare input (X) and output (y)
X = data.drop('Data Accuracy', axis=1)
y = data['Data Accuracy']

# Standardize the data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models to evaluate
models = {
    'LinearRegression': LinearRegression(),
    'DecisionTree': DecisionTreeRegressor(),
    'RandomForest': RandomForestRegressor(),
    'GradientBoosting': GradientBoostingRegressor(),
    'AdaBoost': AdaBoostRegressor(),
    'XGBoost': XGBRegressor(),
    'MLP': MLPRegressor()
}

# Define hyperparameters for grid search
param_grid = {
    'LinearRegression': {},
    'DecisionTree': {'max_depth': [3, 5, 7, 10]},
    'RandomForest': {'n_estimators': [50, 100, 150]},
    'GradientBoosting': {'n_estimators': [50, 100, 150]},
    'AdaBoost': {'n_estimators': [50, 100, 150]},
    'XGBoost': {'n_estimators': [50, 100, 150], 'learning_rate': [0.01, 0.1, 0.2]},
    'MLP': {'hidden_layer_sizes': [(50,), (100,)], 'alpha': [0.0001, 0.001]}
}

# Evaluate models with k-fold cross-validation and grid search
kf = KFold(n_splits=5, shuffle=True, random_state=42)
best_models = {}
model_performance = []

for name, model in models.items():
    print(f'Evaluating {name}...')
    if name in param_grid:
        grid_search = GridSearchCV(model, param_grid[name], cv=kf, scoring='neg_mean_squared_error', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
    else:
        best_model = model
        best_model.fit(X_train, y_train)

    y_pred = best_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    best_models[name] = best_model
    model_performance.append({'Model': name, 'MSE': mse, 'R2': r2})
    print(f'{name} - MSE: {mse}, R2: {r2}')

# Convert model performance to DataFrame
performance_df = pd.DataFrame(model_performance)

# Visualize model performance
plt.figure(figsize=(14, 7))
sns.barplot(x='Model', y='R2', data=performance_df)
plt.title('Model Performance Comparison (R2 Score)')
plt.xlabel('Model')
plt.ylabel('R2 Score')
plt.xticks(rotation=45)
plt.show()

# Visualizing the results of the best model
best_model_name = performance_df.loc[performance_df['R2'].idxmax()]['Model']
best_model = best_models[best_model_name]
y_pred = best_model.predict(X_test)

plt.figure(figsize=(10, 7))
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Data Accuracy')
plt.ylabel('Predicted Data Accuracy')
plt.title(f'Actual vs Predicted Data Accuracy ({best_model_name})')
plt.show()
