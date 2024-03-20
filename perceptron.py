import numpy as np


def step_function(y, threshold=0.5):
    return  np.where(y >= threshold, 1, 0)


def sigmoid(y):
    s = 1 / (1 + np.exp(-y))
    return s


class Perceptron:

    def __init__(self, activation_func, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.activation_func = activation_func
        self.weights = []
        self.bias = None
        self.costs = []

    def cost(self, y, y_pred):
        return np.sum((y - y_pred) ** 2)

    def fit(self, X, y):

        observations, features = X.shape

        self.weights = np.random.rand(features)
        self.bias = 0

        # stochastic gradient
        for i in range(self.num_iterations):
            c = 0
            for xi, yi in zip(X, y):
                y_pred = self.forward_propagation(xi)

                da = 2 * (yi - y_pred) * xi
                db = 2 * (yi - y_pred)

                self.weights = self.weights + self.learning_rate * da
                self.bias = self.bias + self.learning_rate * db

                c += self.cost(yi, y_pred)

            self.costs.append(c)

    def forward_propagation(self, X):
        y = np.dot(X, self.weights) + self.bias
        a = self.activation_func(y)
        return a

    def predict(self, X):
        y = np.dot(X, self.weights) + self.bias
        y_pred = self.activation_func(y)
        y_pred_binary = step_function(y_pred)
        return y_pred_binary

