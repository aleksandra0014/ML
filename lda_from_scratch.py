import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


class LDA:
    def __init__(self, n):
        self.n_components = n
        self.W = None

    def standarization(self, x):
        sc = StandardScaler()
        return sc.fit_transform(x)

    def fit(self, x, y):
        x = self.standarization(x)

        num_features = x.shape[1]
        class_labels = np.unique(y)

        mean_overall = np.mean(x, axis=0)
        S_W = np.zeros((num_features, num_features))
        S_B = np.zeros((num_features, num_features))

        for c in class_labels:
            X_c = x[y == c]
            mean_c = np.mean(X_c, axis=0)
            # (4, n_c) * (n_c, 4) = (4,4)
            S_W += (X_c - mean_c).T.dot(X_c - mean_c)

            n_c = X_c.shape[0]  # num of class
            mean_diff = (mean_c - mean_overall).reshape(num_features, 1)
            S_B += n_c * mean_diff.T.dot(mean_diff)

        A = np.linalg.inv(S_W).dot(S_B)
        eigenvalues, eigenvectors = np.linalg.eig(A)
        eigenvectors = eigenvectors.T
        # argsort : [3,1,2] - > [1, 2, 0]
        idxs = np.argsort(abs(eigenvalues))[::-1]
        eigenvalues = eigenvalues[idxs]
        eigenvectors = eigenvectors[idxs]
        self.W = eigenvectors[0:self.n_components]

    def transform(self, x):
        return np.dot(x, self.W.T)


iris = datasets.load_iris()
X = iris.data
y = iris.target

# PCA
lda = LDA(n=2)
lda.fit(X, y)
X_projected = lda.transform(X)

fig = plt.figure(figsize=(10, 5))

ax1 = fig.add_subplot(121, projection='3d')
ax1.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, marker='o')
ax1.set_title('Original Data')
ax1.set_xlabel('X1')
ax1.set_ylabel('X2')
ax1.set_zlabel('X3')

ax2 = fig.add_subplot(122)
ax2.scatter(X_projected[:, 0], X_projected[:, 1], c=y, marker='o')
ax2.set_title('Transformed Data')
ax2.set_xlabel('Principal Component 1')
ax2.set_ylabel('Principal Component 2')

plt.tight_layout()
plt.show()
