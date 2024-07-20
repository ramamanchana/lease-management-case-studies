
# Explanation
# Sample Data Creation: The script creates a sample dataset containing lease data, metadata, and data accuracy labels. It saves this dataset as lease_data.csv.
# Data Loading and Preprocessing: The script reads the sample data, preprocesses the text by converting it to lowercase, and vectorizes it using TF-IDF.
# Model Definition and Training: A RandomForestClassifier is defined and trained on the TF-IDF vectors to predict data accuracy.
# Model Evaluation: The model's performance is evaluated using accuracy, confusion matrix, and classification report metrics.
# Visualization: The results are visualized using a confusion matrix plot and a feature importance plot.

# Before running the script, ensure you have the necessary libraries installed:
# pip install pandas numpy matplotlib scikit-learn seaborn



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns

# Sample data creation
data = {
    'Lease Data': [
        "Lease agreement for office space at location A with a term of 5 years.",
        "Retail lease for shop B with renewal options and a term of 10 years.",
        "Industrial lease for warehouse C with a term of 3 years and options to extend.",
        "Residential lease for apartment D with a term of 1 year and annual renewal.",
        "Lease for mixed-use building E with various commercial tenants."
    ],
    'Metadata': [
        "Office space, 5 years, Location A",
        "Retail shop, 10 years, Renewal options, Location B",
        "Warehouse, 3 years, Extension options, Location C",
        "Apartment, 1 year, Annual renewal, Location D",
        "Mixed-use building, Commercial tenants, Location E"
    ],
    'Data Accuracy': [1, 1, 0, 1, 0]
}

# Create a DataFrame
data = pd.DataFrame(data)

# Save the sample data to a CSV file
data.to_csv('lease_data.csv', index=False)

# Load the data
data = pd.read_csv('lease_data.csv')

# Data Preprocessing
data['Lease Data'] = data['Lease Data'].str.lower()
data['Metadata'] = data['Metadata'].str.lower()

# Vectorize the text data using TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()
lease_data_tfidf = vectorizer.fit_transform(data['Lease Data'])
metadata_tfidf = vectorizer.fit_transform(data['Metadata'])

# Combine TF-IDF vectors and convert to dense arrays
X = np.hstack((lease_data_tfidf.toarray(), metadata_tfidf.toarray()))
y = data['Data Accuracy']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define and train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
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
plot_confusion_matrix(conf_matrix, classes=['Inaccurate', 'Accurate'])
plt.show()

# Plotting the feature importances
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(12, 6))
plt.title("Feature Importances")
plt.bar(range(X.shape[1]), importances[indices], align="center")
plt.xticks(range(X.shape[1]), indices)
plt.xlim([-1, X.shape[1]])
plt.show()
