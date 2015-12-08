__author__ = 'Juxtux'

import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.datasets.samples_generator import make_blobs

""" Generate sample data: """
# 4 centroids:
centers = [[1, 1], [-1, -1], [1, -1], [-1.5, 3]]
# Data with 15,000 data-points, 4 potential centroids, and specific standard deviation:
X, _ = make_blobs(n_samples=15000, centers=centers, cluster_std=0.6)

# The following bandwidth can be automatically detected using:
bandwidth = estimate_bandwidth(X, quantile=0.1, n_samples=500)
# Clustering with MeanShift and specific bandwidth for the RBF Kernel to use.
ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
ms.fit(X)
labels = ms.labels_
cluster_centers = ms.cluster_centers_       # Get centroids of the model

labels_unique = np.unique(labels)
n_clusters_ = len(labels_unique)

print("number of estimated clusters : %d" % n_clusters_)

""" Plot result: """
import matplotlib.pyplot as plt
from itertools import cycle

plt.figure(1)
plt.clf()       # just clear the current figure

colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')      # Is just a cycle list, in this case with
                                                    # 4 centroids its going to simply use just 'bgrc'
for k, col in zip(range(n_clusters_), colors):
    my_members = labels == k        # simply a boolean vector identifying the clustering labels
    cluster_center = cluster_centers[k]         # centroid of the corresponding label
    plt.plot(X[my_members, 0], X[my_members, 1], col + '.')
    plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col, markeredgecolor='k', markersize=14)
plt.title('Mean-shift Clustering | %d clusters' % n_clusters_)
plt.show()
