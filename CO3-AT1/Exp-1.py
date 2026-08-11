# Experiment 1: Customer Segmentation using K-Means and PCA

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# Load Mall Customers dataset
# Download Mall_Customers.csv and keep it in the same folder
df = pd.read_csv("Mall_Customers.csv")

print(df.head())
print(df.info())

# Select useful numerical features
# Annual Income and Spending Score are commonly used
X = df[["Annual Income (k$)", "Spending Score (1-100)"]]

# Standardize data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------------------------------
# Elbow Method
# -------------------------------

inertia = []

for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(7, 5))
plt.plot(range(2, 11), inertia, marker='o')
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.title("Elbow Method")
plt.show()

# -------------------------------
# Silhouette Score
# -------------------------------

silhouette_scores = []

for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)

    score = silhouette_score(X_scaled, labels)
    silhouette_scores.append(score)

plt.figure(figsize=(7, 5))
plt.plot(range(2, 11), silhouette_scores, marker='o')
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score")
plt.show()

# Select optimal clusters
optimal_k = silhouette_scores.index(max(silhouette_scores)) + 2

print("Optimal number of clusters:", optimal_k)

# -------------------------------
# Apply K-Means
# -------------------------------

kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)

df["Cluster"] = clusters

print("\nClustered Data:")
print(df[["CustomerID", "Cluster"]].head(10))

# -------------------------------
# PCA
# -------------------------------

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print("\nExplained Variance Ratio:")
print(pca.explained_variance_ratio_)

# -------------------------------
# Visualization
# -------------------------------

plt.figure(figsize=(8, 6))

plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=clusters,
    cmap="viridis",
    s=50
)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("Customer Clusters using K-Means + PCA")
plt.colorbar(label="Cluster")

plt.show()
