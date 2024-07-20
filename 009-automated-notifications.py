# Sample Data Creation: The script creates a sample dataset containing lease dates, notification preferences, historical compliance data, and notification alerts. This dataset is saved as lease_notifications.csv.
# Data Loading and Preprocessing: The script reads the sample data, preprocesses the dates by converting them to ordinal format, and encodes categorical data.
# Feature Engineering and Preparation: The script prepares the input features (lease dates, notification preferences, and historical compliance data) and the output target (notification alerts).
# Model Definition and Training: A Decision Tree model is defined and trained to predict notification alerts.
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
    'Lease Dates': pd.date_range(start='1/1/2021', periods=100, freq='M'),
    'Notification Preferences': np.random.choice(['Email', 'SMS', 'None'], size=100),
    'Historical Compliance Data': np.random.randint(0, 2, size=100),
    'Notification Alerts': np.random.randint(0, 2, size=100)  # 0: No Alert, 1: Send Alert
}

# Create DataFrame
data = pd.DataFrame(data)

# Save the sample data to a CSV file
data.to_csv('lease_notifications.csv', index=False)

# Load the data
data = pd.read_csv('lease_notifications.csv', parse_dates=['Lease Dates'])

# Feature Engineering
data['Lease Dates'] = data['Lease Dates'].map(pd.Timestamp.toordinal)

# Encode categorical data
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
data['Notification Preferences'] = label_encoder.fit_transform(data['Notification Preferences'])

# Prepare input (X) and output (y)
X = data[['Lease Dates', 'Notification Preferences', 'Historical Compliance Data']]
y = data['Notification Alerts']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define and train the Decision Tree model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Predicting notifications
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
plot_confusion_matrix(conf_matrix, classes=['No Alert', 'Send Alert'])
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
