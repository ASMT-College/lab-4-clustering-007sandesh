import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4"  # Set this to your physical CPU cores count to silence joblib warning

import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, MiniBatchKMeans, AgglomerativeClustering
from sklearn.datasets import load_iris

# 1. K-means Clustering on 10,000 points (5 clusters)
print("1. K-means Clustering on 10,000 data points")
data1 = np.random.rand(10000, 2) * 100
kmeans1 = KMeans(n_clusters=5, random_state=42, n_init='auto')
start1 = time.time()
kmeans1.fit(data1)
end1 = time.time()
print(f"Time taken by K-means: {end1 - start1:.4f} seconds")
print("Cluster Centers:\n", kmeans1.cluster_centers_)

# 2. Mini-batch K-means with batch size variation
print("\n2. Mini-batch K-means Clustering on 10,000 data points with varying batch sizes")
batch_sizes = [100, 300, 500, 1000, 1500]
mini_times = []
data2 = np.random.rand(10000, 2) * 100

for batch_size in batch_sizes:
    mbk = MiniBatchKMeans(n_clusters=5, batch_size=batch_size, random_state=42, n_init='auto')
    start = time.time()
    mbk.fit(data2)
    end = time.time()
    time_taken = end - start
    mini_times.append(time_taken)
    print(f"Time taken with batch size {batch_size}: {time_taken:.4f} seconds")

best_batch = batch_sizes[np.argmin(mini_times)]
print(f"Best batch size: {best_batch}")

# 3. K-means Clustering on 1,000 points (3 clusters)
print("\n3. K-means Clustering on 1,000 data points with 3 clusters")
data3 = np.random.rand(1000, 2) * 100
kmeans3 = KMeans(n_clusters=3, random_state=42, n_init='auto')
kmeans3.fit(data3)

plt.figure()
plt.scatter(data3[:, 0], data3[:, 1], c=kmeans3.labels_, cmap='viridis')
plt.scatter(kmeans3.cluster_centers_[:, 0], kmeans3.cluster_centers_[:, 1], s=300, c='red')
plt.title("K-means Clustering (3 Clusters)")
plt.show()

# 4. K-means++ Clustering on 1,000 points (4 clusters)
print("\n4. K-means++ Clustering on 1,000 data points with 4 clusters")
data4 = np.random.rand(1000, 2) * 200
kmeans_plus = KMeans(n_clusters=4, init='k-means++', random_state=42, n_init='auto')
kmeans_plus.fit(data4)

plt.figure()
plt.scatter(data4[:, 0], data4[:, 1], c=kmeans_plus.labels_, cmap='viridis')
plt.scatter(kmeans_plus.cluster_centers_[:, 0], kmeans_plus.cluster_centers_[:, 1], s=300, c='red')
plt.title("K-means++ Clustering (4 Clusters)")
plt.show()

# 5. KMedoids on Iris dataset (requires sklearn_extra)
try:
    from sklearn_extra.cluster import KMedoids
    print("\n5. KMedoids Clustering on Iris Dataset")
    iris = load_iris()
    iris_data = iris.data
    kmedoids = KMedoids(n_clusters=3, random_state=42)
    kmedoids.fit(iris_data)

    plt.figure()
    plt.scatter(iris_data[:, 0], iris_data[:, 1], c=kmedoids.labels_, cmap='viridis')
    plt.title("KMedoids Clustering on Iris Dataset")
    plt.show()
except ModuleNotFoundError:
    print("\n5. KMedoids Clustering on Iris Dataset skipped. Please install scikit-learn-extra using 'pip install scikit-learn-extra'")

# 6. Agglomerative Clustering on Iris dataset
print("\n6. Agglomerative Clustering on Iris Dataset")
agg = AgglomerativeClustering(n_clusters=3)
agg_labels = agg.fit_predict(load_iris().data)

plt.figure()
plt.scatter(load_iris().data[:, 0], load_iris().data[:, 1], c=agg_labels, cmap='viridis')
plt.title("Agglomerative Clustering on Iris Dataset")
plt.show()
