#Exploring Data
import pandas as pd
import numpy as np
import random
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.metrics import adjusted_rand_score
from sklearn.utils import resample
import seaborn as sns
from sklearn.mixture import GaussianMixture



n = 100

yes_no = ["Yes", "No"]
like_options = ["-2", "-1", "0", "+1", "+2"]
visit_freq = ["Once a week", "Once a month", "Every three months", "Rarely", "Never"]
gender_options = ["Male", "Female", "Other"]

data = {
    "yummy": np.random.choice(yes_no, n),
    "convenient": np.random.choice(yes_no, n),
    "spicy": np.random.choice(yes_no, n),
    "fattening": np.random.choice(yes_no, n),
    "greasy": np.random.choice(yes_no, n),
    "fast": np.random.choice(yes_no, n),
    "cheap": np.random.choice(yes_no, n),
    "tasty": np.random.choice(yes_no, n),
    "disgusting": np.random.choice(yes_no, n),
    "expensive": np.random.choice(yes_no, n),
    "healthy": np.random.choice(yes_no, n),
    "Like": np.random.choice(like_options, n),
    "Age": np.random.randint(16, 80, n),
    "VisitFrequency": np.random.choice(visit_freq, n),
    "Gender": np.random.choice(gender_options, n)
}


mcdonalds = pd.DataFrame(data)


print(mcdonalds.columns.tolist())
print(mcdonalds.shape)
print(mcdonalds.head(3))
MD_x = mcdonalds.iloc[:, 0:11]


MD_x_numeric = (MD_x == "Yes").astype(int)

column_means = MD_x_numeric.mean().round(2)

print(column_means)
MD_x = mcdonalds.iloc[:, 0:11]
MD_x_binary = (MD_x == "Yes").astype(int)


scaler = StandardScaler()
MD_x_scaled = scaler.fit_transform(MD_x_binary)

pca = PCA()
MD_pca = pca.fit(MD_x_scaled)

explained_variance = pca.explained_variance_**0.5
prop_variance = pca.explained_variance_ratio_
cum_variance = prop_variance.cumsum()

summary_df = pd.DataFrame({
    "Standard Deviation": explained_variance.round(4),
    "Proportion of Variance": prop_variance.round(4),
    "Cumulative Proportion": cum_variance.round(4)
}, index=[f"PC{i+1}" for i in range(len(explained_variance))])

print(summary_df)
MD_x = mcdonalds.iloc[:, 0:11]
MD_x_binary = (MD_x == "Yes").astype(int)
scaler = StandardScaler()
MD_x_scaled = scaler.fit_transform(MD_x_binary)

pca = PCA()
pca.fit(MD_x_scaled)

std_devs = np.sqrt(pca.explained_variance_).round(1)
print("Standard deviations of PCs:")
print(std_devs)

rotation_matrix = pd.DataFrame(
    pca.components_.T.round(2),
    columns=[f'PC{i+1}' for i in range(len(pca.components_))],
    index=MD_x.columns
)
print("\nRotation matrix (loadings):")
print(rotation_matrix)
MD_x = mcdonalds.iloc[:, 0:11]
MD_x_binary = (MD_x == "Yes").astype(int)


scaler = StandardScaler()
MD_x_scaled = scaler.fit_transform(MD_x_binary)

range_clusters = list(range(2, 9))
sil_scores = []
models = []

for n_clusters in range_clusters:
    kmeans = KMeans(n_clusters=n_clusters, random_state=1234, n_init=10)
    kmeans.fit(MD_x_scaled)
    models.append(kmeans)


    score = silhouette_score(MD_x_scaled, kmeans.labels_)
    sil_scores.append(score)


plt.plot(range_clusters, sil_scores, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Scores for Different Numbers of Clusters')
plt.show()


best_n_clusters = range_clusters[np.argmax(sil_scores)]
best_model = models[np.argmax(sil_scores)]


cluster_labels = best_model.labels_

mcdonalds['Cluster'] = cluster_labels


print(mcdonalds.head())
scaler = StandardScaler()
MD_x_scaled = scaler.fit_transform(MD_x_binary)


range_clusters = list(range(2, 9))
sil_scores = []
wcss = []
for n_clusters in range_clusters:
    kmeans = KMeans(n_clusters=n_clusters, random_state=1234, n_init=10)
    kmeans.fit(MD_x_scaled)

    sil_scores.append(silhouette_score(MD_x_scaled, kmeans.labels_))

    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8, 6))

plt.plot(range_clusters, sil_scores, marker='o', color='b', label='Silhouette Score')

plt.plot(range_clusters, wcss, marker='x', color='r', label='WCSS (Elbow Method)')

plt.xlabel('Number of Segments (Clusters)')
plt.ylabel('Score / WCSS')
plt.title('Clustering Evaluation')
plt.legend()
plt.show()
scaler = StandardScaler()
MD_x_scaled = scaler.fit_transform(MD_x_binary)
n_bootstrap = 100
cluster_range = range(2, 9)
ari_values = {n: [] for n in cluster_range}

for n_clusters in cluster_range:

    kmeans = KMeans(n_clusters=n_clusters, random_state=1234, n_init=10)
    kmeans.fit(MD_x_scaled)
    original_labels = kmeans.labels_


    for _ in range(n_bootstrap):

        MD_x_resampled, labels_resampled = resample(MD_x_scaled, original_labels, random_state=1234)


        kmeans_resampled = KMeans(n_clusters=n_clusters, random_state=1234, n_init=10)
        kmeans_resampled.fit(MD_x_resampled)


        ari = adjusted_rand_score(original_labels, kmeans_resampled.labels_)
        ari_values[n_clusters].append(ari)

plt.figure(figsize=(10, 6))
plt.boxplot([ari_values[n] for n in cluster_range], labels=cluster_range)
plt.xlabel('Number of Segments (Clusters)')
plt.ylabel('Adjusted Rand Index')
plt.title('Global Stability of Clustering (Bootstrap Resampling)')
plt.show()
np.random.seed(1234)
n_rows = 1453
columns = ['yummy', 'convenient', 'spicy', 'fattening', 'greasy', 'fast',
           'cheap', 'tasty', 'disgusting', 'expensive', 'healthy']
MD_x_binary = pd.DataFrame(np.random.choice([0, 1], size=(n_rows, 11)), columns=columns)


print("Column names:", MD_x_binary.columns.tolist())
print("Shape:", MD_x_binary.shape)
print("Head:")
print(MD_x_binary.head(3))


scaler = StandardScaler()
MD_x_scaled = scaler.fit_transform(MD_x_binary)
kmeans = KMeans(n_clusters=4, random_state=1234, n_init=10)
labels = kmeans.fit_predict(MD_x_scaled)


MD_x_binary['Cluster'] = labels


features = columns
n_rows_plot = int(np.ceil(len(features) / 3))
fig, axes = plt.subplots(n_rows_plot, 3, figsize=(15, 4 * n_rows_plot), sharex=True)

for i, feature in enumerate(features):
    row = i // 3
    col = i % 3
    ax = axes[row, col] if n_rows_plot > 1 else axes[col]

    sns.histplot(
        data=MD_x_binary,
        x=feature,
        hue='Cluster',
        multiple='stack',
        bins=[-0.5, 0.5, 1.5],
        shrink=0.8,
        palette='tab10',
        ax=ax
    )
    ax.set_xlim(0, 1)
    ax.set_title(f'Feature: {feature}')


for j in range(i + 1, n_rows_plot * 3):
    fig.delaxes(axes[j // 3, j % 3] if n_rows_plot > 1 else axes[j % 3])

plt.tight_layout()
plt.suptitle('Histogram of Binary Features by Cluster (4 Clusters)', fontsize=16, y=1.02)
plt.show()
kmeans_original = KMeans(n_clusters=4, random_state=1234, n_init=10)
original_labels = kmeans_original.fit_predict(MD_x_scaled)


n_bootstraps = 100
n_clusters = 4
segment_ari_scores = [[] for _ in range(n_clusters)]

for i in range(n_bootstraps):

    X_resampled, y_resampled = resample(MD_x_scaled, original_labels, random_state=1234 + i)


    kmeans_boot = KMeans(n_clusters=4, random_state=1234 + i, n_init=10)
    boot_labels = kmeans_boot.fit_predict(X_resampled)


    ari = adjusted_rand_score(y_resampled, boot_labels)


    for j in range(n_clusters):
        segment_ari_scores[j].append(ari)


avg_stability = [np.mean(scores) for scores in segment_ari_scores]


plt.figure(figsize=(8, 5))
plt.bar(range(1, n_clusters + 1), avg_stability, color='skyblue')
plt.ylim(0, 1)
plt.xlabel('Segment number')
plt.ylabel('Segment stability (approx. ARI)')
plt.title('Segment Stability (Bootstrap, 100 reps)')
plt.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.show()
np.random.seed(1234)
n_rows = 1453
columns = ['yummy', 'convenient', 'spicy', 'fattening', 'greasy', 'fast',
           'cheap', 'tasty', 'disgusting', 'expensive', 'healthy']
MD_x_binary = pd.DataFrame(np.random.choice([0, 1], size=(n_rows, len(columns))), columns=columns)

results = []

for k in range(2, 9):
    best_model = None
    best_ll = -np.inf


    for rep in range(10):
        model = GaussianMixture(n_components=k, max_iter=1000, tol=1e-6, random_state=1234 + rep)
        try:
            model.fit(MD_x_binary.values)
            ll = model.score(MD_x_binary.values)
            if ll > best_ll:
                best_ll = ll
                best_model = model
        except Exception as e:
            continue

    if best_model:
        aic = best_model.aic(MD_x_binary.values)
        bic = best_model.bic(MD_x_binary.values)
        icl = bic
        results.append([k, best_model.n_iter_, best_ll, aic, bic, icl])

results_df = pd.DataFrame(results, columns=["k", "iterations", "logLik", "AIC", "BIC", "ICL"])
print(results_df.sort_values("BIC"))
np.random.seed(1234)
n_rows = 1453
columns = ['yummy', 'convenient', 'spicy', 'fattening', 'greasy', 'fast',
           'cheap', 'tasty', 'disgusting', 'expensive', 'healthy']
MD_x_binary = pd.DataFrame(np.random.choice([0, 1], size=(n_rows, len(columns))), columns=columns)


results = []


for k in range(2, 9):
    best_model = None
    best_ll = -np.inf

    for rep in range(10):
        model = GaussianMixture(n_components=k, max_iter=1000, tol=1e-6, random_state=1234 + rep)
        model.fit(MD_x_binary.values)
        ll = model.score(MD_x_binary.values)


        if ll > best_ll:
            best_ll = ll
            best_model = model

    if best_model:
        aic = best_model.aic(MD_x_binary.values)
        bic = best_model.bic(MD_x_binary.values)
        icl = bic
        results.append([k, best_model.n_iter_, best_ll, aic, bic, icl])


results_df = pd.DataFrame(results, columns=["k", "iterations", "logLik", "AIC", "BIC", "ICL"])


plt.figure(figsize=(8, 6))
plt.plot(results_df["k"], results_df["AIC"], label="AIC", marker='o')
plt.plot(results_df["k"], results_df["BIC"], label="BIC", marker='o')
plt.plot(results_df["k"], results_df["ICL"], label="ICL", marker='o')
plt.xlabel("Number of Classes (k)")
plt.ylabel("Information Criteria")
plt.title("Model Selection: AIC, BIC, ICL")
plt.legend()
plt.show()

kmeans = KMeans(n_clusters=4, random_state=1234).fit(MD_x_binary.values)
kmeans_labels = kmeans.labels_


gmm_model_4 = GaussianMixture(n_components=4, random_state=1234)
gmm_model_4.fit(MD_x_binary.values)
gmm_labels = gmm_model_4.predict(MD_x_binary.values)

contingency_table = pd.crosstab(kmeans_labels, gmm_labels, rownames=["kmeans"], colnames=["mixture"])
print(contingency_table)
np.random.seed(1234)
n_rows = 1453
columns = ['yummy', 'convenient', 'spicy', 'fattening', 'greasy', 'fast',
           'cheap', 'tasty', 'disgusting', 'expensive', 'healthy']
MD_x_binary = pd.DataFrame(np.random.choice([0, 1], size=(n_rows, len(columns))), columns=columns)


kmeans = KMeans(n_clusters=4, random_state=1234).fit(MD_x_binary.values)
kmeans_labels = kmeans.labels_

gmm = GaussianMixture(n_components=4, random_state=1234)
gmm_labels = gmm.fit_predict(MD_x_binary.values)


contingency_table = pd.crosstab(kmeans_labels, gmm_labels, rownames=["kmeans"], colnames=["mixture"])


print(contingency_table)

np.random.seed(1234)
n_rows = 1453
columns = ['yummy', 'convenient', 'spicy', 'fattening', 'greasy', 'fast',
           'cheap', 'tasty', 'disgusting', 'expensive', 'healthy']
MD_x_binary = np.random.choice([0, 1], size=(n_rows, len(columns)))


gmm_m4a = GaussianMixture(n_components=4, random_state=1234)
gmm_m4a.fit(MD_x_binary)

gmm_m4 = GaussianMixture(n_components=4, random_state=5678)
gmm_m4.fit(MD_x_binary)

log_lik_m4a = gmm_m4a.score(MD_x_binary) * n_rows
log_lik_m4 = gmm_m4.score(MD_x_binary) * n_rows

print(f"logLik(MD.m4a): {log_lik_m4a} (df=47)")
print(f"logLik(MD.m4): {log_lik_m4} (df=47)")

from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd  # Import pandas for data handling
import numpy as np  # Import numpy for numerical operations

# Create a sample DataFrame (replace with your actual data)
MD_x = pd.DataFrame(np.random.rand(10, 5), columns=['A', 'B', 'C', 'D', 'E'])

# Assuming MD_x is your data matrix of shape (n_samples, n_attributes)
linkage_matrix = linkage(pdist(MD_x.T), method='ward')  # Cluster columns

# Get new attribute order
dendro = dendrogram(linkage_matrix, no_plot=True)
ordered_cols = [MD_x.columns[i] for i in dendro['leaves']]

import matplotlib.pyplot as plt
import numpy as np  # Import numpy if not already imported

# Assuming k4_labels are your segment labels (0, 1, 2, 3)
# Replace this with your actual cluster labels if you have them
# For demonstration, creating random labels here:
k4_labels = np.random.randint(0, 4, size=MD_x.shape[0])

MD_x['Segment'] = k4_labels

# Compute segment profiles
segment_profiles = MD_x.groupby('Segment')[ordered_cols].mean().T

# Plot
segment_profiles.plot(kind='barh', figsize=(10, 6))
plt.title("Segment Profile Plot")
plt.xlabel("Average Score")
plt.legend(title="Segment")
plt.tight_layout()
plt.show()

from sklearn.decomposition import PCA

pca = PCA(n_components=2)
projected = pca.fit_transform(MD_x.drop(columns=['Segment']))

plt.figure(figsize=(8, 6))
sns.scatterplot(x=projected[:, 0], y=projected[:, 1], hue=MD_x['Segment'], palette='tab10')
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("Segment Separation Plot")
plt.legend(title="Segment")
plt.grid(True)
plt.show()

import pandas as pd
from statsmodels.graphics.mosaicplot import mosaic

# Load your data into the 'mcdonalds' DataFrame
# Replace 'mcdonalds_data.csv' with your actual file path, including directory if needed


# Assuming k4_labels are derived from clustering applied to 'mcdonalds' data
# Adjust this to match how you obtain k4_labels for your actual data
# For this example, we'll generate random labels with the same length as 'mcdonalds'
n = 100

yes_no = ["Yes", "No"]
like_options = ["-2", "-1", "0", "+1", "+2"]
visit_freq = ["Once a week", "Once a month", "Every three months", "Rarely", "Never"]
gender_options = ["Male", "Female", "Other"]

data = {
    "yummy": np.random.choice(yes_no, n),
    "convenient": np.random.choice(yes_no, n),
    "spicy": np.random.choice(yes_no, n),
    "fattening": np.random.choice(yes_no, n),
    "greasy": np.random.choice(yes_no, n),
    "fast": np.random.choice(yes_no, n),
    "cheap": np.random.choice(yes_no, n),
    "tasty": np.random.choice(yes_no, n),
    "disgusting": np.random.choice(yes_no, n),
    "expensive": np.random.choice(yes_no, n),
    "healthy": np.random.choice(yes_no, n),
    "Like": np.random.choice(like_options, n),
    "Age": np.random.randint(16, 80, n),
    "VisitFrequency": np.random.choice(visit_freq, n),
    "Gender": np.random.choice(gender_options, n)
}


mcdonalds = pd.DataFrame(data)

k4_labels = np.random.randint(0, 4, size=mcdonalds.shape[0])

# Contingency table
contingency_like = pd.crosstab(k4_labels, mcdonalds['Like'])

# Mosaic plot
mosaic(contingency_like.stack(), title='Segment vs Like')
plt.xlabel("Segment Number")
plt.show()
contingency_gender = pd.crosstab(k4_labels, mcdonalds['Gender'])
mosaic(contingency_gender.stack(), title='Segment vs Gender')
plt.xlabel("Segment Number")
plt.show()
sns.boxplot(x=k4_labels, y=mcdonalds['Age'], notch=True)
plt.xlabel("Segment Number")
plt.ylabel("Age")
plt.title("Age by Segment")
plt.grid(True)
plt.show()

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder

# Create target: whether consumer belongs to segment 3
target = (k4_labels == 3).astype(int)

# Features
# Changed 'Like.n' to 'Like' assuming it's the correct column name
features = mcdonalds[['Like', 'Age', 'VisitFrequency', 'Gender']]  # Gender may need encoding

# Create a LabelEncoder object
le = LabelEncoder()

# Fit and transform the 'Like' column to numerical values
features['Like'] = le.fit_transform(features['Like'])

# Encode 'Gender' and 'VisitFrequency' as well
features['Gender'] = features['Gender'].astype('category').cat.codes
features['VisitFrequency'] = le.fit_transform(features['VisitFrequency'])


tree = DecisionTreeClassifier(criterion='entropy', max_depth=3)
tree.fit(features, target)

plt.figure(figsize=(12, 6))
plot_tree(tree, feature_names=features.columns, class_names=['Not Segment 3', 'Segment 3'], filled=True)
plt.title("Classification Tree for Segment 3")
plt.show()
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Assume k4_labels is your segment membership array (0 to 3)
mcdonalds['Segment'] = k4_labels

# Create a LabelEncoder object
le = LabelEncoder()

# Fit and transform the 'VisitFrequency' and 'Like' columns to numerical values
mcdonalds['VisitFrequency'] = le.fit_transform(mcdonalds['VisitFrequency'])
mcdonalds['Like'] = le.fit_transform(mcdonalds['Like'])


# Group by segment and calculate metrics
# Changed 'Like.n' to 'Like' in the aggregation dictionary
eval_df = mcdonalds.groupby('Segment').agg({
    'VisitFrequency': 'mean',
    'Like': 'mean',  # Corrected column name
    'Gender': lambda x: (x == 'Female').mean() * 100  # Percent female
}).reset_index()

# Rename columns for clarity
eval_df.columns = ['Segment', 'MeanVisitFrequency', 'MeanLike', 'PercentFemale']

# Plot
plt.figure(figsize=(10, 6))
plt.scatter(
    eval_df['MeanVisitFrequency'],
    eval_df['MeanLike'],
    s=eval_df['PercentFemale'] * 10,  # Scale bubble size
    alpha=0.6,
    c=eval_df['Segment'],
    cmap='tab10',
    edgecolors='black'
)

for i, row in eval_df.iterrows():
    plt.text(row['MeanVisitFrequency'] + 0.02, row['MeanLike'], f"Segment {int(row['Segment'])}", fontsize=10)

plt.xlabel("Visit Frequency (Mean)")
plt.ylabel("Like Score (Mean)")
plt.title("Segment Evaluation Plot")
plt.grid(True)
plt.tight_layout()
plt.show()
target_segment = 3
target_df = mcdonalds[mcdonalds['Segment'] ==target_segment]
summary = {
    "Segment": target_segment,
    "Size (%)": round(len(target_df) / len(mcdonalds) * 100, 1),
    "Avg Age": round(target_df["Age"].mean(), 1),
    "Gender (% Female)": round((target_df["Gender"] == "Female").mean() * 100, 1),
    "Avg Visit Frequency": round(target_df["VisitFrequency"].mean(), 2),
    "Avg Like Score": round(target_df["Like"].mean(), 2)  # Changed 'Like.n' to 'Like'
}
pd.DataFrame([summary])
