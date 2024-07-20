# Explanation
# Sample Data Creation: The script creates a sample dataset containing lease portfolio data with lease ID, lease cost, lease area, market value, business priority, market condition, and disposition recommendation. This dataset is saved as lease_portfolio_disposition.csv.
# Data Loading and Preprocessing: The script reads the sample data from the CSV file and encodes the categorical data (market condition).
# Feature Engineering and Preparation: The script prepares the input features (lease cost, lease area, market value, business priority, and market condition) and the output target (disposition recommendation).
# Model Definition and Training: A Decision Tree model is defined and trained to predict disposition recommendations.
# Model Evaluation: The model's performance is evaluated using accuracy, confusion matrix, and classification report metrics.
# Visualization: The results are visualized using a confusion matrix heatmap, a plot of the Decision Tree, and a feature importance bar plot.
# Before running the script, ensure you have the necessary libraries installed:

# pip install pandas numpy matplotlib seaborn scikit-learn


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns

# Sample data creation
data = {
    'Lease ID': ['L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7', 'L8', 'L9', 'L10'],
    'Lease Cost': np.random.randint(1000, 5000, size=10),
    'Lease Area': np.random.randint(500, 2000, size=10),
    'Market Value': np.random.randint(2000, 6000, size=10),
    'Business Priority': np.random.randint(1, 10, size=10),
    'Market Condition': np.random.choice(['Favorable', 'Neutral', 'Unfavorable'], size=10),
    'Disposition Recommendation': np.random.randint(0, 2, size=10)  # 0: Keep, 1: Dispose
}

# Create DataFrame
data = pd.DataFrame(data)

# Save the sample data to a CSV file
data.to_csv('lease_portfolio_disposition.csv', index=False)

# Load the data
data = pd.read_csv('lease_portfolio_disposition.csv')

# Encode categorical data
data['Market Condition'] = data['Market Condition'].astype('category').cat.codes

# Prepare input (X) and output (y)
X = data[['Lease Cost', 'Lease Area', 'Market Value', 'Business Priority', 'Market Condition']]
y = data['Disposition Recommendation']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define and train the Decision Tree model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Predicting disposition recommendations
y_pred = model.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print('Confusion Matrix:')
print(conf_matrix)
print('Classification Report:')
print(class_report)

# Visualizing the results
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Keep', 'Dispose'], yticklabels=['Keep', 'Dispose'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Plotting the Decision Tree
plt.figure(figsize=(20, 10))
plot_tree(model, feature_names=X.columns, class_names=['Keep', 'Dispose'], filled=True, rounded=True)
plt.title('Decision Tree for Lease Disposition Recommendations')
plt.show()

# Feature Importance
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

# Get feature names
feature_names = X.columns

# Determine the number of features to plot
num_features = len(feature_names)

plt.figure(figsize=(12, 6))
plt.title("Feature Importances for Disposition Recommendations")
plt.bar(range(num_features), importances[indices], align="center")
plt.xticks(range(num_features), [feature_names[i] for i in indices], rotation=90)
plt.xlim([-1, num_features])
plt.show()
