# Explanation
# Sample Data Creation: The script creates a sample dataset containing historical lease data, payment records, expense reports, cost savings opportunities, and recoverable amounts. This dataset is saved as lease_audits.csv.
# Data Loading and Preprocessing: The script reads the sample data and preprocesses the text using TF-IDF.
# Feature Engineering and Preparation: The script prepares the input features (TF-IDF vectors of historical lease data, payment records, and expense reports) and the output targets (cost savings opportunities and recoverable amounts).
# Model Definition and Training: Two Decision Tree models are defined and trained, one for cost savings opportunities and another for recoverable amounts.
# Model Evaluation: The models' performances are evaluated using accuracy, confusion matrix, and classification report metrics.
# Visualization: The results are visualized using confusion matrix plots and feature importance plots, with the number of features to plot dynamically adjusted based on the total number of features available.
# Before running the script, ensure you have the necessary libraries installed:

# pip install pandas numpy matplotlib seaborn scikit-learn



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
import seaborn as sns

# Sample data creation
data = {
    'Historical Lease Data': [
        "Lease for office space with monthly payment of $1000.",
        "Lease for retail space with monthly payment of $1500.",
        "Lease for warehouse with monthly payment of $2000.",
        "Lease for apartment with monthly payment of $1200.",
        "Lease for mixed-use building with monthly payment of $1700."
    ],
    'Payment Records': [
        "$1000 payment for office lease.",
        "$1500 payment for retail lease.",
        "$2500 payment for warehouse lease.",
        "$1200 payment for apartment lease.",
        "$1700 payment for mixed-use lease."
    ],
    'Expense Reports': [
        "Office lease expenses are $12000 annually.",
        "Retail lease expenses are $18000 annually.",
        "Warehouse lease expenses are $24000 annually.",
        "Apartment lease expenses are $14400 annually.",
        "Mixed-use lease expenses are $20400 annually."
    ],
    'Cost Savings Opportunities': [0, 0, 1, 0, 0],  # 1: Opportunity, 0: No Opportunity
    'Recoverable Amounts': [0, 0, 500, 0, 0]  # Recoverable amount in dollars
}

# Create DataFrame
data = pd.DataFrame(data)

# Save the sample data to a CSV file
data.to_csv('lease_audits.csv', index=False)

# Load the data
data = pd.read_csv('lease_audits.csv')

# Data Preprocessing
# Vectorize the text data using TF-IDF
lease_vectorizer = TfidfVectorizer()
payment_vectorizer = TfidfVectorizer()
expense_vectorizer = TfidfVectorizer()

lease_data_tfidf = lease_vectorizer.fit_transform(data['Historical Lease Data'])
payment_records_tfidf = payment_vectorizer.fit_transform(data['Payment Records'])
expense_reports_tfidf = expense_vectorizer.fit_transform(data['Expense Reports'])

# Combine TF-IDF vectors and convert to dense arrays
X = np.hstack((lease_data_tfidf.toarray(), payment_records_tfidf.toarray(), expense_reports_tfidf.toarray()))
y_opportunities = data['Cost Savings Opportunities']
y_recoverable = data['Recoverable Amounts']

# Train-test split
X_train_op, X_test_op, y_train_op, y_test_op = train_test_split(X, y_opportunities, test_size=0.2, random_state=42)
X_train_rec, X_test_rec, y_train_rec, y_test_rec = train_test_split(X, y_recoverable, test_size=0.2, random_state=42)

# Define and train the Decision Tree models
model_opportunities = DecisionTreeClassifier(random_state=42)
model_opportunities.fit(X_train_op, y_train_op)

model_recoverable = DecisionTreeClassifier(random_state=42)
model_recoverable.fit(X_train_rec, y_train_rec)

# Predicting cost savings opportunities
y_pred_op = model_opportunities.predict(X_test_op)
y_pred_rec = model_recoverable.predict(X_test_rec)

# Evaluation for cost savings opportunities
accuracy_op = accuracy_score(y_test_op, y_pred_op)
conf_matrix_op = confusion_matrix(y_test_op, y_pred_op)
class_report_op = classification_report(y_test_op, y_pred_op)

print(f'Accuracy for Cost Savings Opportunities: {accuracy_op}')
print('Confusion Matrix for Cost Savings Opportunities:')
print(conf_matrix_op)
print('Classification Report for Cost Savings Opportunities:')
print(class_report_op)

# Evaluation for recoverable amounts
accuracy_rec = accuracy_score(y_test_rec, y_pred_rec)
conf_matrix_rec = confusion_matrix(y_test_rec, y_pred_rec)
class_report_rec = classification_report(y_test_rec, y_pred_rec)

print(f'Accuracy for Recoverable Amounts: {accuracy_rec}')
print('Confusion Matrix for Recoverable Amounts:')
print(conf_matrix_rec)
print('Classification Report for Recoverable Amounts:')
print(class_report_rec)

# Visualizing the results
def plot_confusion_matrix(cm, classes, title='Confusion Matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = 'd'
    thresh = cm.max() / 2
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

plt.figure(figsize=(10, 7))
plot_confusion_matrix(conf_matrix_op, classes=['No Opportunity', 'Opportunity'])
plt.show()

plt.figure(figsize=(10, 7))
plot_confusion_matrix(conf_matrix_rec, classes=['No Recovery', 'Recovery'])
plt.show()

# Feature Importance for cost savings opportunities
importances_op = model_opportunities.feature_importances_
indices_op = np.argsort(importances_op)[::-1]

# Get feature names from all vectorizers
lease_features = lease_vectorizer.get_feature_names_out()
payment_features = payment_vectorizer.get_feature_names_out()
expense_features = expense_vectorizer.get_feature_names_out()

feature_names_op = np.concatenate([lease_features, payment_features, expense_features])

# Determine the number of features to plot
num_features_op = min(20, len(importances_op))

plt.figure(figsize=(12, 6))
plt.title("Feature Importances for Cost Savings Opportunities")
plt.bar(range(num_features_op), importances_op[indices_op[:num_features_op]], align="center")
plt.xticks(range(num_features_op), [feature_names_op[i] for i in indices_op[:num_features_op]], rotation=90)
plt.xlim([-1, num_features_op])
plt.show()

# Feature Importance for recoverable amounts
importances_rec = model_recoverable.feature_importances_
indices_rec = np.argsort(importances_rec)[::-1]

# Determine the number of features to plot
num_features_rec = min(20, len(importances_rec))

plt.figure(figsize=(12, 6))
plt.title("Feature Importances for Recoverable Amounts")
plt.bar(range(num_features_rec), importances_rec[indices_rec[:num_features_rec]], align="center")
plt.xticks(range(num_features_rec), [feature_names_op[i] for i in indices_rec[:num_features_rec]], rotation=90)
plt.xlim([-1, num_features_rec])
plt.show()
