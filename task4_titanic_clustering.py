# ============================================================
# TASK 4: TITANIC UNSUPERVISED LEARNING
# K-Means Clustering and PCA
# ============================================================

# 1. Import required libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


# ============================================================
# 2. LOAD CLEANED DATASET
# ============================================================

data = pd.read_csv("titanic_cleaned.csv")

print("=" * 60)
print("TITANIC UNSUPERVISED LEARNING")
print("=" * 60)

print("\nCleaned dataset loaded successfully!")

print("\nDataset Shape:")
print(data.shape)

print("\nFirst 5 rows:")
print(data.head())


# ============================================================
# 3. PREPARE DATA
# ============================================================

# Remove the target variable because this is
# an unsupervised learning task.

X = data.drop("Survived", axis=1)

print("\nNumber of features used for clustering:",
      X.shape[1])


# ============================================================
# 4. FEATURE SCALING
# ============================================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print("\nFeature scaling completed.")


# ============================================================
# 5. K-MEANS CLUSTERING
# ============================================================

print("\n" + "=" * 60)
print("K-MEANS CLUSTERING")
print("=" * 60)

# Create K-Means model with 3 clusters

kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

# Train the model and assign clusters

clusters = kmeans.fit_predict(X_scaled)

# Add cluster labels to the dataset

data["Cluster"] = clusters

print("\nK-Means clustering completed!")

print("\nCluster Distribution:")
print(
    data["Cluster"].value_counts().sort_index()
)


# ============================================================
# 6. PCA DIMENSIONALITY REDUCTION
# ============================================================

print("\n" + "=" * 60)
print("PRINCIPAL COMPONENT ANALYSIS (PCA)")
print("=" * 60)

# Reduce the features to 2 principal components

pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_scaled)

print("\nPCA completed successfully!")

print("\nExplained Variance Ratio:")

print(
    "PC1:",
    f"{pca.explained_variance_ratio_[0]:.4f}"
)

print(
    "PC2:",
    f"{pca.explained_variance_ratio_[1]:.4f}"
)

total_variance = (
    pca.explained_variance_ratio_[0]
    + pca.explained_variance_ratio_[1]
)

print(
    "\nTotal Explained Variance:",
    f"{total_variance * 100:.2f}%"
)


# ============================================================
# 7. VISUALIZE K-MEANS CLUSTERS USING PCA
# ============================================================

plt.figure(figsize=(9, 6))

plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=clusters,
    s=40
)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")

plt.title(
    "Titanic K-Means Clusters using PCA"
)

plt.colorbar(
    label="Cluster"
)

plt.show()


# ============================================================
# 8. DISPLAY CLUSTER CENTERS IN PCA SPACE
# ============================================================

# Transform K-Means cluster centers into PCA space

cluster_centers_pca = pca.transform(
    kmeans.cluster_centers_
)

plt.figure(figsize=(9, 6))

plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=clusters,
    s=40
)

plt.scatter(
    cluster_centers_pca[:, 0],
    cluster_centers_pca[:, 1],
    marker="X",
    s=200,
    label="Cluster Centers"
)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")

plt.title(
    "K-Means Clusters and Cluster Centers"
)

plt.legend()

plt.show()


# ============================================================
# 9. SAVE CLUSTERED DATASET
# ============================================================

data.to_csv(
    "titanic_clustered.csv",
    index=False
)

print("\nClustered dataset saved as:")
print("titanic_clustered.csv")


# ============================================================
# 10. FINAL RESULT
# ============================================================

print("\n" + "=" * 60)
print("TASK 4 COMPLETED SUCCESSFULLY!")
print("=" * 60)

print(
    "\nNumber of clusters:",
    kmeans.n_clusters
)

print(
    "PCA components:",
    pca.n_components_
)

print(
    "Explained variance:",
    f"{total_variance * 100:.2f}%"
)