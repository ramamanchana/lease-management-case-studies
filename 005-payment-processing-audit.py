# Explanation
# Sample Data Creation: The script creates a sample dataset containing payment records, lease terms, financial statements, and payment accuracy labels. This dataset is saved as payment_processing.csv.
# Data Loading and Preprocessing: The script reads the sample data, preprocesses the text using TF-IDF, and encodes categorical data.
# Feature Engineering and Preparation: The script prepares the input features (TF-IDF vectors of payment records, lease terms, and financial statements) and the output target (payment accuracy).
# Model Definition and Training: A Random Forest model is defined and trained to detect anomalies in payment records.
# Model Evaluation: The model's performance is evaluated using accuracy, confusion matrix, and classification report metrics.
# Visualization: The results are visualized using a confusion matrix plot and a feature importance plot.

# Before running the script, ensure you have the necessary libraries installed:

# pip install pandas numpy matplotlib seaborn scikit-learn


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import seaborn as sns

# Sample data creation
data = {
    'Payment Records': [
        "Payment of $1000 for lease term A",
        "Payment of $1500 for lease term B",
        "Payment of $2000 for lease term C",
        "Payment of $1200 for lease term A",
        "Payment of $1700 for lease term B"
    ],
    'Lease Terms': [
        "Term A: $1000 per month",
        "Term B: $1500 per month",
        "Term C: $2000 per month",
        "Term A: $1000 per month",
        "Term B: $1500 per month"
    ],
    'Financial Statements': [
        "Financial statement 1",
        "Financial statement 2",
        "Financial statement 3",
        "Financial statement 4",
        "Financial statement 5"
    ],
    'Payment Accuracy': [1, 1, 1, 0, 0]  # 1: Accurate, 0: Discrepancy
}

# Create DataFrame
data = pd.DataFrame(data)

# Save the sample data to a CSV file
data.to_csv('payment_processing.csv', index=False)

# Load the data
data = pd.read_csv('payment_processing.csv')

# Data Preprocessing
# Vectorize the text data using TF-IDF
vectorizer = TfidfVectorizer()
payment_records_tfidf = vectorizer.fit_transform(data['Payment Records'])
lease_terms_tfidf = vectorizer.fit_transform(data['Lease Terms'])
financial_statements_tfidf = vectorizer.fit_transform(data['Financial Statements'])

# Combine TF-IDF vectors and convert to dense arrays
X = np.hstack((payment_records_tfidf.toarray(), lease_terms_tfidf.toarray(), financial_statements_tfidf.toarray()))
y = data['Payment Accuracy']

# Encode categorical data
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define and train the Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predicting payment accuracy
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
plot_confusion_matrix(conf_matrix, classes=['Discrepancy', 'Accurate'])
plt.show()

# Feature Importance
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]
features = vectorizer.get_feature_names_out()

plt.figure(figsize=(12, 6))
plt.title("Feature Importances")
plt.bar(range(20), importances[indices[:20]], align="center")
plt.xticks(range(20), [features[i] for i in indices[:20]], rotation=90)
plt.xlim([-1, 20])
plt.show()
