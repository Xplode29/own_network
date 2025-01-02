import numpy as np
import matplotlib.pyplot as plt

# Fonction d'activation (sigmoïde) et sa dérivée
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - x ** 2

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def leaky_relu_derivative(x, alpha=0.01):
    return np.where(x > 0, 1, alpha)

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def mse_derivative(y_true, y_pred):
    return (y_pred - y_true) / y_true.size

class Layer:
    """Représente une couche d'un réseau de neurones."""
    def __init__(self, input_size, output_size, activation=sigmoid, activation_derivative=sigmoid_derivative):
        self.weights = np.random.rand(input_size, output_size) - 0.5
        self.biases = np.random.rand(output_size) - 0.5
        self.output = None
        self.input = None
        self.delta = None
        
        self.activation = activation
        self.activation_derivative = activation_derivative

    def forward(self, x):
        """Passe avant : calcule les sorties."""
        self.input = x
        self.output = self.activation(np.dot(x, self.weights) + self.biases)
        return self.output

    def backward(self, gradient, learning_rate):
        """Passe arrière : met à jour les poids et biais."""
        self.delta = gradient * self.activation_derivative(self.output)
        weight_gradient = np.dot(self.input.T, self.delta)
        self.weights -= learning_rate * weight_gradient
        self.biases -= learning_rate * np.sum(self.delta, axis=0)
        
        # Retourne l'erreur à propager à la couche précédente
        return np.dot(self.delta, self.weights.T)

class NeuralNetwork:
    """Classe pour gérer un réseau de neurones complet."""
    def __init__(self, input_size):
        self.input_size = input_size
        self.layers : list[Layer] = []

    def add_layer(self, output_size, activation=sigmoid, activation_derivative=sigmoid_derivative):
        """Ajoute une couche au réseau."""
        if len(self.layers) == 0:
            layer = Layer(self.input_size, output_size, activation, activation_derivative)
        else:
            layer = Layer(self.layers[-1].weights.shape[1], output_size, activation, activation_derivative)
        self.layers.append(layer)

    def forward(self, x):
        """Passe avant à travers toutes les couches."""
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, y_true, y_pred, learning_rate):
        """Passe arrière pour rétropropager l'erreur."""
        # Erreur initiale pour la couche de sortie
        gradient_output = y_pred - y_true
        for layer in reversed(self.layers):
            gradient_output = layer.backward(gradient_output, learning_rate)

    def train(self, X, Y, epochs, learning_rate):
        """Entraîne le réseau sur des données données."""
        for epoch in range(epochs):
            # Passe avant
            Y_pred = self.forward(X)

            # Calcul de l'erreur
            loss = np.mean((Y - Y_pred) ** 2)

            # Passe arrière
            self.backward(Y, Y_pred, learning_rate)

            # Affichage périodique de la perte
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.5f}")

    def predict(self, X):
        """Prédit les sorties pour de nouvelles données."""
        return self.forward(X)
