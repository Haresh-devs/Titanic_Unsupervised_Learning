Titanic Unsupervised Learning
📌 Project Overview

This task applies unsupervised machine learning techniques to the cleaned and feature-engineered Titanic dataset.

Two techniques are used:

K-Means Clustering – to group similar passenger records.
Principal Component Analysis (PCA) – to reduce the dimensionality of the dataset and visualize the clusters.
🎯 Objectives
Use the cleaned Titanic dataset from Task 2.
Prepare the data for unsupervised learning.
Apply feature scaling.
Perform K-Means clustering.
Divide the data into 3 clusters.
Apply PCA to reduce the feature space to 2 dimensions.
Visualize the resulting clusters.
Save the clustered dataset.
📊 Dataset

The cleaned Titanic dataset contains:

Item	Result
Rows	891
Columns	32
Features used for clustering	31
Target excluded	Survived

The Survived column was excluded because K-Means is an unsupervised learning algorithm and should not use the known target label when forming clusters.

⚙️ Methodology
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
🔄 Data Preprocessing

The cleaned and feature-engineered dataset generated in Task 2 was used.

The Survived column was removed before clustering:

X = data.drop("Survived", axis=1)

This resulted in 31 features being used for unsupervised learning.

The features were standardized using:

StandardScaler()
🔵 K-Means Clustering

K-Means clustering was applied with:

Number of clusters = 3
Random state = 42
Cluster Distribution
Cluster	Number of Records
Cluster 0	80
Cluster 1	287
Cluster 2	524

The cluster labels were added to the dataset as a new Cluster column.

📉 Principal Component Analysis

PCA was used to reduce the 31-dimensional feature space to 2 dimensions.

The two principal components were:

Component	Explained Variance
PC1	13.28%
PC2	9.78%
Total	23.07%

The first two principal components together explain 23.07% of the total variance in the standardized dataset.

📊 Visualization

Two visualizations were generated:

1. K-Means Clusters using PCA

The passengers are plotted using:

X-axis → Principal Component 1
Y-axis → Principal Component 2
Different colors → K-Means clusters
2. K-Means Clusters and Cluster Centers

This visualization additionally displays the three K-Means cluster centers in the PCA-reduced space.

📁 Project Structure
Task_4/
│
├── titanic_cleaned.csv
├── task4_titanic_clustering.py
├── titanic_clustered.csv
└── README.md

🛠️ Technologies Used
Python
Pandas – Data loading and manipulation
NumPy – Numerical operations
Matplotlib – Data visualization
Scikit-learn – StandardScaler, K-Means, and PCA
🚀 How to Run
1. Install the required libraries
pip install pandas numpy matplotlib scikit-learn
2. Make sure the files are in the same folder
titanic_cleaned.csv
task4_titanic_clustering.py
3. Run the program
python task4_titanic_clustering.py

The program will:

Load the cleaned dataset.
Remove the Survived target.
Scale the features.
Perform K-Means clustering.
Display the cluster distribution.
Apply PCA.
Display explained variance.
Generate the cluster visualizations.
Save the clustered dataset.
📈 Final Results
Number of clusters: 3

Cluster 0: 80
Cluster 1: 287
Cluster 2: 524

PCA components: 2

PC1: 13.28%
PC2: 9.78%

Total explained variance: 23.07%

The final clustered dataset is saved as:

titanic_clustered.csv
📝 Conclusion

K-Means clustering was successfully applied to the cleaned Titanic dataset and divided the observations into three clusters.

PCA successfully reduced the 31-dimensional feature space to two principal components, which were used to visualize the clusters.

The first two principal components explained 23.07% of the total variance. The complete clustering results were saved in titanic_clustered.csv.

👨‍💻 Author

Haresh K

Computer Science and Engineering
Sri Krishna College of Engineering and Technology

Internship: AIML & Data Science Internship
Organization: Edufiy
