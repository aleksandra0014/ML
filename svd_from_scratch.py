from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
class SVD:
    def __init__(self, k):
        self.k = k

    def standarization(self, x):
        sc = StandardScaler()
        return sc.fit_transform(x)

    def calU(self, X):
        B = np.dot(X, X.T)
        eigenvalues, eigenvectors = np.linalg.eig(B)
        ncols = np.argsort(eigenvalues)[::-1]
        return eigenvectors[:, ncols]

    def colVt(self, X):
        B = np.dot(X.T, X)
        eigenvalues, eigenvectors = np.linalg.eig(B)
        ncols = np.argsort(eigenvalues)[::-1]
        return eigenvectors[:, ncols].T

    def calS(self, X):
        if (np.size(np.dot(X, X.T)) > np.size(np.dot(X.T, X))):
            newM = np.dot(X.T, X)
        else:
            newM = np.dot(X, X.T)
        eigenvalues, eigenvectors = np.linalg.eig(newM)
        eigenvalues = np.sqrt(eigenvalues)
        # Sorting in descending order
        return eigenvalues[::-1]

    def svd(self, X):

        X = self.standarization(X)

        U = self.calU(X)
        Vt = self.colVt(X)
        S = self.calS(X)
        S = np.diag(S)

        return U[:, :self.k] @ S[:self.k, :self.k] @ Vt[:self.k, :]


s = SVD(2)
iris = datasets.load_iris()
X = iris.data
Y = iris.target

principal_components = s.svd(X)
setosa_pc = principal_components[iris.target == 0]
versicolor_pc = principal_components[iris.target == 1]
virginica_pc = principal_components[iris.target == 2]

plt.figure(figsize=(8, 6))
plt.scatter(setosa_pc[:, 0], setosa_pc[:, 1], label='Setosa')
plt.scatter(versicolor_pc[:, 0], versicolor_pc[:, 1], label='Versicolor')
plt.scatter(virginica_pc[:, 0], virginica_pc[:, 1], label='Virginica')

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Principal Component Analysis of Iris dataset')
plt.legend()
plt.show()