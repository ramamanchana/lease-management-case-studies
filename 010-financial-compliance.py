# Case Study: Financial Compliance Monitoring in Real Estate Lease Abstraction

"""
Description:
Monitoring financial transactions to ensure compliance with lease terms and regulatory standards. This involves using ML models to analyze financial data and identify potential compliance issues.

Model: Predictive Analytics - Decision Trees

Data Input: Financial Data, Lease Terms, Regulatory Standards

Prediction: Compliance Status, Anomalies

Recommended Model: Decision Trees for analyzing financial data and ensuring compliance

Customer Value Benefits: Regulatory Compliance, Risk Mitigation

Use Case Implementation:
By using decision trees to analyze financial data, stakeholders can ensure compliance with lease terms and regulatory standards, identify potential compliance issues, and mitigate risks.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# Sample data creation
np.random.seed(42)
financial_data = np.random.rand(100, 5)  # Dummy financial data
lease_terms = np.random.randint(1, 5, size=(100, 1))  # Dummy lease terms
regulatory_standards = np.random.randint(1, 5, size=(100, 1))  # Dummy regulatory standards
compliance_status = np.random.choice([0, 1], size=100)  # 0: Non-compliant, 1: Compliant

# Create DataFrame
data = pd.DataFrame(np.hstack((financial_data, lease_terms, regulatory_standards)), columns=[f'Feature_{i}' for i in range(7)])
data['Compliance Status'] = compliance_status

# Save the sample data to a CSV file
data.to_csv('financial_compliance_monitoring.csv', index=False)

# Load the data
data = pd.read_csv('financial_compliance_monitoring.csv')

# Prepare input (X) and output (y)
X = data.drop('Compliance Status', axis=1)
y = data['Compliance Status']

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Define Decision Tree model
model = DecisionTreeClassifier()

# Define hyperparameters for grid search
param_grid = {
    'max_depth': [3, 5, 7, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Grid search with cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
grid_search = GridSearchCV(model, param_grid, cv=kf, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

# Make predictions
y_pred = best_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f'Best Model: {best_model}')
print(f'Accuracy: {accuracy}')
print(f'Confusion Matrix:\n{conf_matrix}')
print(f'Classification Report:\n{class_report}')

# Visualizing the confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-compliant', 'Compliant'], yticklabels=['Non-compliant', 'Compliant'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Feature Importance
importances = best_model.feature_importances_
indices = np.argsort(importances)[::-1]
feature_names = [f'Feature_{i}' for i in range(7)]

plt.figure(figsize=(12, 6))
plt.title("Feature Importances")
plt.bar(range(len(feature_names)), importances[indices], align="center")
plt.xticks(range(len(feature_names)), [feature_names[i] for i in indices], rotation=90)
plt.xlim([-1, len(feature_names)])
plt.show()
