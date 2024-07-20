# Explanation
# Sample Data Creation: The script now creates a larger sample dataset containing 50 entries of expense data, lease terms, historical expense reports, and cost savings opportunities. This dataset is saved as expense_analysis.csv.
# Data Loading and Preprocessing: The script reads the sample data and preprocesses the lease terms using TF-IDF.
# Feature Engineering and Preparation: The script prepares the input features (expense data, historical expense reports, and TF-IDF vectors of lease terms) and the output target (cost savings opportunities).
# Model Definition and Training: A Linear Regression model is defined and trained to predict cost savings opportunities.
# Model Evaluation: The model's performance is evaluated using mean squared error and R-squared metrics. With a larger dataset, the test set should have enough samples to define the R-squared score.
# Visualization: The results are visualized using a scatter plot of actual vs predicted values and a feature importance plot, with the number of features to plot dynamically adjusted based on the total number of features available.
# Before running the script, ensure you have the necessary libraries installed:

# pip install pandas numpy matplotlib seaborn scikit-learn


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_extraction.text import TfidfVectorizer
import seaborn as sns

# Sample data creation
data = {
    'Expense Data': np.random.randint(10000, 30000, size=50),
    'Lease Terms': [
        "Lease for office space with monthly payment of $1000.",
        "Lease for retail space with monthly payment of $1500.",
        "Lease for warehouse with monthly payment of $2000.",
        "Lease for apartment with monthly payment of $1200.",
        "Lease for mixed-use building with monthly payment of $1700."
    ] * 10,
    'Historical Expense Reports': np.random.randint(9000, 29000, size=50),
    'Cost Savings Opportunities': np.random.randint(0, 1000, size=50)  # Savings amount in dollars
}

# Create DataFrame
data = pd.DataFrame(data)

# Save the sample data to a CSV file
data.to_csv('expense_analysis.csv', index=False)

# Load the data
data = pd.read_csv('expense_analysis.csv')

# Encode Lease Terms
vectorizer = TfidfVectorizer()
lease_terms_tfidf = vectorizer.fit_transform(data['Lease Terms'])

# Combine all features
X = np.hstack((data[['Expense Data', 'Historical Expense Reports']].values, lease_terms_tfidf.toarray()))
y = data['Cost Savings Opportunities']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define and train the Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predicting cost savings opportunities
y_pred = model.predict(X_test)

# Evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

# Visualizing the results
plt.figure(figsize=(10, 7))
plt.scatter(y_test, y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted Cost Savings Opportunities')
plt.show()

# Feature Importance
coefficients = model.coef_
indices = np.argsort(np.abs(coefficients))[::-1]

# Get feature names from all vectorizers
expense_features = ['Expense Data', 'Historical Expense Reports']
lease_features = vectorizer.get_feature_names_out()

feature_names = np.concatenate([expense_features, lease_features])

# Determine the number of features to plot
num_features = min(20, len(coefficients))

plt.figure(figsize=(12, 6))
plt.title("Feature Importances for Cost Savings Opportunities")
plt.bar(range(num_features), coefficients[indices[:num_features]], align="center")
plt.xticks(range(num_features), [feature_names[i] for i in indices[:num_features]], rotation=90)
plt.xlim([-1, num_features])
plt.show()
