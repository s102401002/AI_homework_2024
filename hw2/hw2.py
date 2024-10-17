import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import kneighbors_graph

np.random.seed(0)
X = np.random.rand(100, 2)

n_neighbors = 2
knn = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree')
knn.fit(X)

distances, indices = knn.kneighbors(X)

print(indices)