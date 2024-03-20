import numpy as np
import matplotlib.pyplot as plt


def sigmoid(y):
    return 1 / (1 + np.exp(-y))


def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


def step_function(y, threshold=0.5):
    return  np.where(y >= threshold, 1, 0)


class MLP:
    def __init__(self,activation_func, activation_derivative, num_iterations=100000, learning_rate=1e-1, input_layer=2, hidden_layer=3, output_layer=1):
        self.iterations = num_iterations
        self.learning_rate = learning_rate
        self.activation_func = activation_func
        self.activation_derivative = activation_derivative
        self.input_layer = input_layer
        self.hidden_layer = hidden_layer
        self.output_layer = output_layer
        self.weights_1 = np.random.randn(self.input_layer, self.hidden_layer)
        self.weights_2 = np.random.randn(self.hidden_layer, self.output_layer)
        self.costs = []

    def forward_propagation(self, X):
        z = np.dot(X, self.weights_1)
        h = self.activation_func(z)
        zh = np.dot(h, self.weights_2)
        p = self.activation_func(zh)
        return z, h, zh, p

    def cost(self, y, p):
        return -(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)) / len(y))

    def back_propagation(self, X, y, z, h, zh, p):
        dwh = np.dot(h.T, (p - y))
        d1 = np.multiply((p - y), self.weights_2.T)
        d2 = np.multiply(d1, self.activation_derivative(z))
        dw = np.dot(X.T, d2)
        return dwh, dw

    def train(self, X, y):
        for i in range(self.iterations):
            z, h, zh, p = self.forward_propagation(X)
            dwh, dw = self.back_propagation(X, y, z, h, zh, p)
            self.weights_1 -= self.learning_rate * dw
            self.weights_2 -= self.learning_rate * dwh
            self.costs.append(self.cost(y, p))

    def predict(self, x):
        z, h, zh, y_pred = self.forward_propagation(x)
        y_pred_binary = step_function(y_pred)
        return y_pred_binary

    def plot_cost_func(self):
        plt.grid()
        plt.plot(range(self.iterations), self.costs)
        plt.xlabel('Iterations')
        plt.ylabel('Cost')
        plt.title(' Cost Function')
        plt.show()


# Przykład działania na bramkach logicznych
X = np.array([[1, 0], [0, 1], [0, 0], [1, 1]])
Y = np.array([[1], [1], [0], [0]])

mlp = MLP(sigmoid, sigmoid_derivative)
mlp.train(X, Y)
print(mlp.predict([1, 0]))
mlp.plot_cost_func()



