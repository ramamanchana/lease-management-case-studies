# Explanation
#  Sample Data Creation: The script creates a sample dataset containing lease document text, accounting standards, historical compliance data, and compliance status. This dataset is saved as compliance_verification.csv.
#  Data Loading and Preprocessing: The script reads the sample data, preprocesses the text using TF-IDF, and encodes categorical data.
#   Feature Engineering and Preparation: The script prepares the input features (TF-IDF vectors of lease document text, accounting standards, and historical compliance data) and the output target (compliance status).
#  Model Definition and Training: A Decision Tree model is defined and trained to verify compliance.
#  Model Evaluation: The model's performance is evaluated using accuracy, confusion matrix, and classification report metrics.
#  Visualization: The results are visualized using a confusion matrix plot and a feature importance plot, with the number of features to plot dynamically adjusted based on the total number of features available.
#  Before running the script, ensure you have the necessary libraries installed:

# pip install pandas numpy matplotlib seaborn scikit-learn


import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import seaborn as sns

# Sample data creation
data = {
    'Lease Document Text': [
        "Lease agreement states payment of $1000 monthly for office space.",
        "Lease terms include $1500 monthly payment for retail space.",
        "Industrial lease requires $2000 monthly for warehouse.",
        "Residential lease states $1200 monthly for apartment.",
        "Lease agreement for mixed-use building at $1700 monthly."
    ],
    'Accounting Standards': [
        "Standard 1: $1000 per month for office",
        "Standard 2: $1500 per month for retail",
        "Standard 3: $2000 per month for industrial",
        "Standard 4: $1200 per month for residential",
        "Standard 5: $1700 per month for mixed-use"
    ],
    'Historical Compliance Data': [
        "Compliant",
        "Compliant",
        "Compliant",
        "Non-compliant",
        "Non-compliant"
    ],
    'Compliance Status': [1, 1, 1, 0, 0]  # 1: Compliant, 0: Non-compliant
}

# Create DataFrame
data = pd.DataFrame(data)

# Save the sample data to a CSV file
data.to_csv('compliance_verification.csv', index=False)

# Load the data
data = pd.read_csv('compliance_verification.csv')

# Encode compliance data
label_encoder = LabelEncoder()
data['Historical Compliance Data'] = label_encoder.fit_transform(data['Historical Compliance Data'])

# Vectorize the text data using TF-IDF
vectorizer = TfidfVectorizer()
lease_text_tfidf = vectorizer.fit_transform(data['Lease Document Text'])
standards_tfidf = vectorizer.fit_transform(data['Accounting Standards'])

# Combine TF-IDF vectors and convert to dense arrays
X = np.hstack((lease_text_tfidf.toarray(), standards_tfidf.toarray(), data[['Historical Compliance Data']].values))
y = data['Compliance Status']

# Train-test split
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
features = vectorizer.get_feature_names_out()

# Determine the number of features to plot
num_features = min(20, len(importances))

plt.figure(figsize=(12, 6))
plt.title("Feature Importances")
plt.bar(range(num_features), importances[indices[:num_features]], align="center")
plt.xticks(range(num_features), [features[i] for i in indices[:num_features]], rotation=90)
plt.xlim([-1, num_features])
plt.show()
