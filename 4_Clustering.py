import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
from matplotlib import cm
from matplotlib.colors import ListedColormap

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.utils import check_random_state

# =================== Loading dataset ======================
file_path = "data/air_quality.csv"
if not os.path.exists(file_path):
    raise FileNotFoundError(f"CSV non trovato -> percorso {file_path}")

df = pd.read_csv(file_path, nrows=1000000, na_values=["-", "NA", "N/A", "", "NaN"])


# ================= Data cleaning ===================
print(df.info())

df.head()
df = df.drop(columns=["sitename", "county", "aqi", "siteid", "pollutant", "date"])
df = df.dropna(axis=1, how="all")
df = df.dropna(axis=1, how="any")

print(df["status"].unique())

print("Different values:", df['status'].nunique())

df_numeric = df.select_dtypes(include=[np.number])
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_numeric)


# ===================== Scelta del numero di cluster ========================
scores = {}

for k in range(2, 6):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(df_scaled)
    score = silhouette_score()
    scores[k] = score
    print(f"Silhouette socre for k={k}: {score}")

best_k = max(scores, key=scores.get)
print(f"\nAccording to the silhoutte analysis, the optimal number of clusters would be: k={best_k}")


# =========================== Clustering k-means ============================
num_clusters = df['status'].nunique()
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
labels = kmeans.fit_predict(df_scaled)


# ========================== Comparison =============================
ct = pd.crosstab(df['status'], labels, colnames=['cluster'])
print("Contingency: status vs cluster")
print(ct)

cluster_to_status = ct.idxmax(axis=0).to_dict()
print("Mapping cluster --> status:", cluster_to_status)

pred_status = pd.Series(labels, index=df.index).map(cluster_to_status)

is_correct = pred_status.eq(df['status'])

summary = (
    pd.DataFrame({'status': df['status'], 'correct': is_correct})
      .groupby('status')['correct']
      .agg(correct='sum', tot='count')
)

print("\nSummary by status (correct/incorrect/accuracy):")
print(summary.sort_values('accuracy_%', ascending=False))

overall_acc = is_correct.mean()


# ============================= PCA Visualization =========================
# 2D visualization with PCA
# Fit PCA on the scaled features and project data to 2 principal components
pca = PCA(n_components=2)
reduced = pca.fit_transform(df_scaled)

# Select the base colormap "tab10"
base_cmap = plt.colormaps["tab10"]

# Take only the first 'num_clusters' colors
colors = base_cmap.colors[:num_clusters]

# Create a discrete colormap with these selected colors
cmap = ListedColormap(colors)

# Scatter plot of the 2D projection, colored by KMeans cluster labels
plt.figure(figsize=(8,6))
sc1 = plt.scatter(reduced[:,0], reduced[:,1], c=labels, cmap=cmap, alpha=0.6, s=12)

# Get the unique cluster IDs found by KMeans
unique_clusters = np.unique(labels)

# Create a colored patch for each cluster
colors = [cmap(i) for i in range(num_clusters)]
patches = [mpatches.Patch(color=colors[i], label=f'Cluster {cl}')
           for i, cl in enumerate(unique_clusters)]

# Add the legend to the plot
plt.legend(handles=patches, title="Labels", loc='lower left', frameon=True)

# Compute cluster sizes to display in the title
sizes = np.bincount(labels)
sizes_txt = ", ".join(f"{i}:{sizes[i]}" for i in range(len(sizes)))

# Title with k, variance explained by PCs, and cluster sizes
plt.title(f'KMeans — PCA 2D (k={num_clusters}) | size [{sizes_txt}]')

plt.tight_layout()
plt.show()


# ============================ Visualization of Real Distribution ===========================
status_labels = df['status'].replace({'Moderate': 0, 'Good': 1, 'Unhealthy for Sensitive Groups': 2, 'Unhealthy': 3, 'Very Unhealthy': 4}, inplace=False).infer_objects(copy=False)

# Scatter plot of the 2D projection, colored by KMeans cluster labels
plt.figure(figsize=(8,6))
sc1 = plt.scatter(reduced[:,0], reduced[:,1], c=status_labels, cmap=cmap, alpha=0.6, s=12)

# Create a colored patch for each cluster
colors = [cmap(i) for i in range(num_clusters)]
patches = [mpatches.Patch(color=colors[i], label=cl)
           for i, cl in enumerate(df['status'].unique())]

# Add the legend to the plot
plt.legend(handles=patches, title="Labels", loc='lower left', frameon=True)

# Title with k, variance explained by PCs, and cluster sizes
plt.title(f'Labels distribution')

plt.tight_layout()
plt.show()


# ============== Comparison Between Cluster and True Label ============
# Color points by correctness: 'orange' if cluster --> status mapping matches the true status, else 'red'
colors = np.where(is_correct.values, 'orange', 'red')

#  2D PCA scatter colored by match/mismatch
plt.figure(figsize=(8,6))
plt.scatter(reduced[:,0], reduced[:,1], c=colors, alpha=0.6, s=12)

# Legend using colored patches
patches = [
    mpatches.Patch(color='orange', label='Match cluster=status'),
    mpatches.Patch(color='red', label='Mismatch')
]
plt.legend(handles=patches, title="Labels", loc='lower left', frameon=True)

# Title shows overall accuracy computed earlier; axis labels include variance explained by PCs
plt.title(f'PCA comparison — Match (orange) vs Mismatch (red) | Acc {overall_acc*100:.1f}%')

plt.tight_layout()
plt.show()


# ======================= Cluster Analysis =====================
# Attach cluster assignments to the original data for interpretation
df_with_clusters = df.copy()
df_with_clusters["Cluster"] = labels

# Compute per-cluster averages
# This summarizes each cluster's centroid in the original feature space,
# which is easier to interpret than scaled values.
cluster_summary = df_with_clusters.groupby('Cluster').mean(numeric_only=True)

# Display the table
print(cluster_summary)