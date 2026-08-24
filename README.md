# Titanic Unsupervised Learning

## Project Overview

This project applies unsupervised machine learning techniques to the cleaned and feature-engineered Titanic dataset.

The following techniques are used:

- K-Means Clustering
- Principal Component Analysis (PCA)

K-Means is used to identify groups of similar passenger records, while PCA is used to reduce the dimensionality of the dataset and visualize the clusters.

## Objectives

- Use the cleaned Titanic dataset.
- Prepare the dataset for unsupervised learning.
- Apply feature scaling.
- Perform K-Means clustering.
- Divide the dataset into 3 clusters.
- Apply PCA to reduce the feature space to 2 dimensions.
- Visualize the clusters.
- Save the clustered dataset.

## Dataset

The cleaned Titanic dataset contains:

| Item | Value |
|---|---:|
| Rows | 891 |
| Columns | 32 |
| Features used for clustering | 31 |
| Excluded column | `Survived` |

The `Survived` column is excluded because K-Means is an unsupervised learning algorithm and does not require the target variable.

## Methodology

```text
Cleaned Titanic Dataset
        ↓
Remove Survived
        ↓
Feature Scaling
        ↓
K-Means Clustering
        ↓
3 Clusters
        ↓
PCA
        ↓
2 Principal Components
        ↓
Cluster Visualization


## Data Preprocessing

The cleaned and feature-engineered Titanic dataset generated in Task 2 is used as the input.

The `Survived` column is removed before applying clustering:

```python
X = data.drop("Survived", axis=1)

After removing the target column, 31 features are used for clustering.

The features are standardized using StandardScaler.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
K-Means Clustering

K-Means clustering is applied using 3 clusters.
Number of clusters = 3
Random state = 42

### Cluster Distribution

| Cluster | Number of Records |
|---|---:|
| Cluster 0 | 80 |
| Cluster 1 | 287 |
| Cluster 2 | 524 |

The cluster assignments are added to the dataset using a new `Cluster` column.

## Principal Component Analysis (PCA)

PCA is used to reduce the 31-dimensional feature space to 2 principal components for visualization.

### Explained Variance

| Component | Explained Variance |
|---|---:|
| PC1 | 13.28% |
| PC2 | 9.78% |
| Total | 23.07% |

The first two principal components together explain **23.07%** of the total variance.

## Visualization

The project generates two visualizations:

### K-Means Clusters using PCA

The PCA-reduced data is plotted using:

- X-axis: Principal Component 1
- Y-axis: Principal Component 2
- Different colors: K-Means clusters

### K-Means Clusters and Cluster Centers

A second graph displays the three K-Means cluster centers along with the clustered passenger data.

## Final Results

```text
Number of clusters: 3

Cluster 0: 80
Cluster 1: 287
Cluster 2: 524

PCA components: 2

PC1: 13.28%
PC2: 9.78%

Total explained variance: 23.07%

The final clustered dataset is saved as:

```text
titanic_clustered.csv

Technologies Used
Python
Pandas
NumPy
Matplotlib
Scikit-learn
Libraries Used
pandas – Data loading and manipulation
numpy – Numerical operations
matplotlib – Data visualization
StandardScaler – Feature scaling
KMeans – Clustering
PCA – Dimensionality reduction
How to Run
1. Install the Required Libraries
pip install pandas numpy matplotlib scikit-learn
2. Make Sure the Following Files Are in the Same Folder
titanic_cleaned.csv
task4_titanic_clustering.py
3. Run the Program
python task4_titanic_clustering.py

The program will:

Load the cleaned Titanic dataset.
Remove the Survived column.
Scale the features.
Apply K-Means clustering.
Display the cluster distribution.
Apply PCA.
Display the explained variance.
Generate cluster visualizations.
Save the clustered dataset.
Project Structure
Task_4/
│
├── titanic_cleaned.csv
├── task4_titanic_clustering.py
├── titanic_clustered.csv
├── Task_4_Titanic_Unsupervised_Learning_Report.pdf
└── README.md
Conclusion

K-Means clustering was successfully applied to the cleaned Titanic dataset and divided the observations into three clusters.

PCA reduced the 31-dimensional feature space to two principal components for visualization. The first two components explained 23.07% of the total variance.

The final clustered dataset was successfully generated and saved as titanic_clustered.csv.

Author

Haresh K

Computer Science and Engineering

Sri Krishna College of Engineering and Technology

Internship: AIML & Data Science Internship

Organization: Edufiy
