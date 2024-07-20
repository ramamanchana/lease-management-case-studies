# Explanation
# Sample Data Creation: The script creates a sample dataset containing market data, lease terms, and benchmarking data. This dataset is saved as benchmarking_analysis.csv.
# Data Loading and Preprocessing: The script reads the sample data and preprocesses the lease terms using TF-IDF.
# Feature Engineering and Preparation: The script prepares the input features (market data, benchmarking data, and TF-IDF vectors of lease terms) and standardizes them.
# Model Definition and Training: A KMeans clustering model is defined and trained to identify clusters of competitive lease terms.
# Model Evaluation: The model's performance is evaluated using the silhouette score to measure cluster quality.
# Visualization: The results are visualized using a scatter plot of the clusters and a bar plot of the cluster centers, with the feature names labeled.
# Before running the script, ensure you have the necessary libraries installed:

# pip install pandas numpy matplotlib seaborn scikit-learn


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.feature_extraction.text import TfidfVectorizer
import seaborn as sns

# Sample data creation
data = {
    'Market Data': np.random.randint(1000, 5000, size=50),
    'Lease Terms': [
        "Lease for office space with a term of 5 years.",
        "Lease for retail space with a term of 3 years.",
        "Lease for warehouse with a term of 7 years.",
        "Lease for apartment with a term of 1 year.",
        "Lease for mixed-use building with a term of 10 years."
    ] * 10,
    'Benchmarking Data': np.random.randint(1000, 5000, size=50)
}

# Create DataFrame
data = pd.DataFrame(data)

# Save the sample data to a CSV file
data.to_csv('benchmarking_analysis.csv', index=False)

# Load the data
data = pd.read_csv('benchmarking_analysis.csv')

# Encode Lease Terms
vectorizer = TfidfVectorizer()
lease_terms_tfidf = vectorizer.fit_transform(data['Lease Terms'])

# Combine all features
X = np.hstack((data[['Market Data', 'Benchmarking Data']].values, lease_terms_tfidf.toarray()))

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Define and train the KMeans model
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_scaled)

# Predicting the clusters
clusters = kmeans.predict(X_scaled)

# Evaluation
silhouette_avg = silhouette_score(X_scaled, clusters)
print(f'Silhouette Score: {silhouette_avg}')

# Add clusters to the DataFrame for visualization
data['Cluster'] = clusters

# Visualizing the clusters
plt.figure(figsize=(10, 7))
sns.scatterplot(data=data, x='Market Data', y='Benchmarking Data', hue='Cluster', palette='viridis')
plt.title('Clusters of Lease Terms')
plt.show()

# Visualizing cluster centers
cluster_centers = kmeans.cluster_centers_

# Inverse transform the scaled centers to original scale
cluster_centers_orig = scaler.inverse_transform(cluster_centers)

# Feature Importance - Here we'll consider the distance to cluster center as importance
for i, center in enumerate(cluster_centers_orig):
    print(f'Cluster {i} center: {center}')

# Get feature names from all vectorizers
market_features = ['Market Data', 'Benchmarking Data']
lease_features = vectorizer.get_feature_names_out()

feature_names = np.concatenate([market_features, lease_features])

plt.figure(figsize=(12, 6))
plt.title("Cluster Centers")
for i in range(len(cluster_centers)):
    plt.bar(range(len(cluster_centers_orig[i])), cluster_centers_orig[i], align="center", alpha=0.5, label=f'Cluster {i}')
plt.xticks(range(len(cluster_centers_orig[0])), feature_names, rotation=90)
plt.legend()
plt.show()
