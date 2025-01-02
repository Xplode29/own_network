from neuralnetwork import *
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles

def test_checkboard():
    N = 1000
    X = np.random.rand(N, 2)
    Y = ((X[:, 0] > 0.5) ^ (X[:, 1] > 0.5)).astype(int).reshape(-1, 1)
    
    # Crée un réseau de neurones
    nn = NeuralNetwork(2)
    nn.add_layer(8, leaky_relu, leaky_relu_derivative)
    nn.add_layer(1)
    nn.train(X, Y, epochs=10000, learning_rate=0.01)
    
    predictions = nn.predict(X)
    plt.scatter(X[:, 0], X[:, 1], c=np.round(predictions.ravel()), cmap="coolwarm")
    plt.title("Prédictions sur un damier")
    plt.show()

def test_circles():
    X, Y = make_circles(n_samples=500, noise=0.1, factor=0.5)
    Y = Y.reshape(-1, 1)
    
    # Crée un réseau de neurones
    nn = NeuralNetwork(2)
    nn.add_layer(16, leaky_relu, leaky_relu_derivative)
    nn.add_layer(1, sigmoid, sigmoid_derivative)
    nn.train(X, Y, epochs=200, learning_rate=0.01)  # Try a lower learning rate 
       
    predictions = nn.predict(X)
    plt.scatter(X[:, 0], X[:, 1], c=np.round(predictions.ravel()), cmap="coolwarm")
    plt.title("Prédictions après entraînement")
    plt.show()

def test_sinus():
    # Crée un réseau de neurones
    nn = NeuralNetwork(1)
    nn.add_layer(10, tanh, tanh_derivative)
    nn.add_layer(1, tanh, tanh_derivative)
    
    X = np.linspace(0, 2 * np.pi, 100).reshape(-1, 1)
    Y = np.sin(X)
    nn.train(X, Y, epochs=10000, learning_rate=0.005)
    
    predictions = nn.predict(X)
    plt.plot(X, Y, label="True function")
    plt.plot(X, predictions, label="Predictions")
    plt.legend()
    plt.show()

def test_xor():
    # Données d'entraînement (fonction XOR)
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    Y = np.array([[0], [1], [1], [0]])

    # Création du réseau
    nn = NeuralNetwork(2)
    nn.add_layer(3, sigmoid, sigmoid_derivative)  # Couche cachée avec 3 neurones
    nn.add_layer(1, sigmoid, sigmoid_derivative)  # Couche de sortie avec 1 neurone

    # Entraînement
    nn.train(X, Y, epochs=1000, learning_rate=1)
    
    # Test de prédiction
    N = 100
    x = np.linspace(0, 1, N)
    y = np.linspace(0, 1, N)
    X_test = np.meshgrid(x, y)

    final_output = nn.predict(np.array(X_test).T.reshape(-1, 2))

    # Affichage de la prédiction
    plt.figure()
    plt.title("Prédiction du modèle")
    plt.contourf(x, y, final_output.reshape(N, N), levels=100, cmap="coolwarm")
    plt.colorbar()
    plt.show()

if __name__ == "__main__":
    #test_checkboard()
    #test_circles()
    test_sinus()
    #test_xor()