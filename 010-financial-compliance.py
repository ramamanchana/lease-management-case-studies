
# Explanation
# Sample Data Creation: The script creates a sample dataset containing financial data, lease terms, regulatory standards, and compliance status. This dataset is saved as financial_compliance.csv.
# Data Loading and Preprocessing: The script reads the sample data and encodes categorical data.
# Feature Engineering and Preparation: The script prepares the input features (financial data, lease terms, and regulatory standards) and the output target (compliance status).
# Model Definition and Training: A Decision Tree model is defined and trained to predict compliance status.
# Model Evaluation: The model's performance is evaluated using accuracy, confusion matrix, and classification report metrics.
# Visualization: The results are visualized using a confusion matrix plot and a feature importance plot.
# Before running the script, ensure you have the necessary libraries installed:
# pip install pandas numpy matplotlib seaborn scikit-learn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns

# Sample data creation
data = {
    'Financial Data': np.random.normal(loc=1000, scale=200, size=100),
    'Lease Terms': np.random.choice(['Term A', 'Term B', 'Term C'], size=100),
    'Regulatory Standards': np.random.choice(['Standard 1', 'Standard 2', 'Standard 3'], size=100),
    'Compliance Status': np.random.randint(0, 2, size=100)  # 0: Non-compliant, 1: Compliant
}

# Create DataFrame
data = pd.DataFrame(data)

# Save the sample data to a CSV file
data.to_csv('financial_compliance.csv', index=False)

# Load the data
data = pd.read_csv('financial_compliance.csv')

# Encode categorical data
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
data['Lease Terms'] = label_encoder.fit_transform(data['Lease Terms'])
data['Regulatory Standards'] = label_encoder.fit_transform(data['Regulatory Standards'])

# Prepare input (X) and output (y)
X = data[['Financial Data', 'Lease Terms', 'Regulatory Standards']]
y = data['Compliance Status']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define and train the Decision Tree model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Predicting compliance status
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
plot_confusion_matrix(conf_matrix, classes=['Non-compliant', 'Compliant'])
plt.show()

# Feature Importance
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]
features = X.columns

plt.figure(figsize=(12, 6))
plt.title("Feature Importances")
plt.bar(range(X.shape[1]), importances[indices], align="center")
plt.xticks(range(X.shape[1]), [features[i] for i in indices])
plt.xlim([-1, X.shape[1]])
plt.show()
