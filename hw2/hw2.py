import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import kneighbors_graph

np.random.seed(0)
X = np.random.rand(100, 2)

n_neighbors = 3
knn = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree')
knn.fit(X)

distances, indices = knn.kneighbors(X)

sparse = knn.kneighbors_graph(X)

print("Nearest neighbors indices:\n", indices)
print("Sparse :\n", sparse.toarray()[0:5])

plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], color='b', label='Data points')

for i in range(X.shape[0]):
    for j in indices[i]:
        plt.plot([X[i, 0], X[j, 0]], [X[i, 1], X[j, 1]], 'k--', lw=0.5)

plt.title('k-Nearest Neighbors Graph (k='+ str(n_neighbors)+ ')')
plt.xlabel('X1')
plt.ylabel('X2')
plt.legend()
plt.show()