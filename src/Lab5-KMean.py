import numpy as np

# Class Initialization
class Kmeans:
    def __init__(self, k=2, max_iter=100):
        self.k = k
        self.max_iter = max_iter
        self.centroids = []
        self.cluster_assignments = 0

    # Build training method
    def fit(self, X):
        N,D = X.shape
        # Select the centers of clusters Randomly
        self.centroids = X[np.random.choice(N, self.k, replace=False), :]

        for i in range(self.max_iter):
            distance = np.zeros((N, self.k))
            for j in range(self.k):
                # Calc the distance between center point with every point
                distance[:, j] = np.sum((X - self.centroids[j, :]) ** 2, axis=1)

            cluster_assignments = np.argmin(distance, axis=1)
            for j in range(self.k):
                # Calc new center from the distances
                self.centroids[j, :] = np.mean(X[cluster_assignments == j, :], axis=0)
        self.cluster_assignments = cluster_assignments

        return self

    # Build testing method
    def predict(self, X):
        N, D = X.shape
        distances = np.zeros((N, self.k))

        for j in range(self.k):     # Calc distance between X (new data) and Centers
            distances[:, j] = np.sum((X - self.centroids[j, :]) ** 2, axis=1)

        return np.argmin(distances,axis=1)


X = np.array([[1, 2], [3, 4], [5, 6], [11, 15], [10, 13], [19, 17]])

model = Kmeans(k=2)
model.fit(X)
predictions = model.predict(X)
print(predictions)
