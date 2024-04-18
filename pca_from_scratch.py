import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets

class PCA:

    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None

    def standarization(self, X):
        return (X - np.mean(X, axis=0)) / np.std(X, axis=0)

    def fit(self, X):
        X_standardized = self.standarization(X)

        cov_matrix = np.cov(X_standardized.T)

        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

        sorted_indices = np.argsort(eigenvalues)[::-1]
        sorted_eigenvalues = eigenvalues[sorted_indices]
        sorted_eigenvectors = eigenvectors[:, sorted_indices]

        self.components = sorted_eigenvectors[:, :self.n_components]

        # Odwróć kierunek wektorów własnych wzdłuż drugiego składowego
        # Odwrócenie kierunku wektora własnego w przypadku analizy głównych składowych (PCA) jest kwestią konwencji
        # wizualizacyjnej i interpretacyjnej.W PCA wektor własny odpowiada kierunkowi maksymalnej wariancji w danych.
        # Celem odwrócenia kierunku wektora własnego wzdłuż drugiej składowej jest zachowanie spójności interpretacji
        # wizualnej z intuicją geometryczną.Domyślnie, biblioteki do wizualizacji (takie jak Matplotlib) przyjmują, że
        # osie rosną w górę na wykresach. Jednak w przypadku analizy głównych składowych, jeśli druga składowa została
        # odwrócona, zgodnie z konwencją, warto odwrócić także kierunek wektora własnego, aby zachować spójność
        # interpretacji. Innymi słowy, chcemy, aby kierunek wektora własnego wizualnie odzwierciedlał kierunek osi na
        # wykresie, co ułatwia zrozumienie znaczenia każdej składowej.

        self.components[:, 1] *= -1

    def transform(self, X):
        # Project data onto principal components
        X_standardized = self.standarization(X)
        return np.dot(X_standardized, self.components)

iris = datasets.load_iris()
X = iris.data
y = iris.target

# PCA
pca = PCA(n_components=2)
pca.fit(X)
X_projected = pca.transform(X)

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
