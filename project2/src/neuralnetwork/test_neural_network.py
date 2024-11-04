from neural_network import NeuralNetwork
import numpy as np
from activation_functions import sigmoid, ReLU, leakyReLU, softmax

def test_layers(nn:NeuralNetwork, input_size:int, layer_sizes:list) -> None:
    """
    This function validates the shapes of the weights and biases for each layer of a neural network.
    It checks if the dimensions match the expected sizes based on the input size and the layer sizes provided.
    Assumes:
    1. Neural network layers are fully connected (dense).
    2. Layers are returned as (weights, biases) tuples by the nn.get_layers() method.
    3. Biases are represented as vectors, and weights as matrices.
    """

    layers = nn.layers

    excpected_layer_sizes_weights = [(input_size, layer_sizes[0])]
    excpected_layer_sizes_biases = [(layer_sizes[0],)]

    n_layers = len(layer_sizes)

    if n_layers > 1:
        for i in range(n_layers):
            excpected_layer_sizes_weights.append((layer_sizes[i], layer_sizes[i+1]))
            excpected_layer_sizes_biases.append((layer_sizes[i+1],))
            if i+2 == n_layers:
                break

    for i, (W, b) in enumerate(layers):

        assert W.shape == excpected_layer_sizes_weights[i], \
            f"Layer {i}: Expected weight shape {excpected_layer_sizes_weights[i]}, but got {W.shape}."
        
        assert b.shape == excpected_layer_sizes_biases[i], \
            f"Layer {i}: Expected bias shape ({excpected_layer_sizes_biases[i]}), but got {b.shape}."
        
    print("All layers have the expected sizes!")

def perform_test_layers():
# Test Set 1: Simple single-layer network
    input_size = 3
    layer_sizes = [4]
    nn = NeuralNetwork(input_size, layer_sizes, [None], loss='mse')
    test_layers(nn, input_size, layer_sizes)

    # Test Set 2: Two-layer network
    input_size = 5
    layer_sizes = [6, 3]
    nn = NeuralNetwork(input_size, layer_sizes, [None, None], loss='mse')
    test_layers(nn, input_size, layer_sizes)

    # Test Set 3: Three-layer network
    input_size = 8
    layer_sizes = [10, 7, 4]
    nn = NeuralNetwork(input_size, layer_sizes, [None, None, None], loss='mse')
    test_layers(nn, input_size, layer_sizes)

    # Test Set 4: Deep network with multiple layers
    input_size = 10
    layer_sizes = [20, 15, 10, 5]
    nn = NeuralNetwork(input_size, layer_sizes, [None, None, None, None], loss='mse')
    test_layers(nn, input_size, layer_sizes)

    # Test Set 5: Large input size with fewer layers
    input_size = 50
    layer_sizes = [30, 20]
    nn = NeuralNetwork(input_size, layer_sizes, [None, None], loss='mse')
    test_layers(nn, input_size, layer_sizes)

    # Test Set 6: Edge case - Input size of 1, single layer
    input_size = 1
    layer_sizes = [1]
    nn = NeuralNetwork(input_size, layer_sizes, [None], loss='mse')
    test_layers(nn, input_size, layer_sizes)

    # Test Set 7: No hidden layers (expect assertion error)
    input_size = 5
    layer_sizes = []  # No hidden layers, should trigger an assertion error
    try:
        nn = NeuralNetwork(input_size, layer_sizes, [], loss='mse')
    except ValueError as e:
        print(f'Correctly handled! {e}')

    # Test Set 8: Larger layers in a deep network
    input_size = 12
    layer_sizes = [64, 32, 16, 8]
    nn = NeuralNetwork(input_size, layer_sizes, [None, None, None, None], loss='mse')
    test_layers(nn, input_size, layer_sizes)

    # Test Set 9: Large input and output, with one intermediate layer
    input_size = 128
    layer_sizes = [64, 128]
    nn = NeuralNetwork(input_size, layer_sizes, [None, None], loss='mse')
    test_layers(nn, input_size, layer_sizes)

    # Test Set 10: Minimal input with multiple layers
    input_size = 2
    layer_sizes = [4, 3, 2]
    nn = NeuralNetwork(input_size, layer_sizes, [None, None, None], loss='mse')
    test_layers(nn, input_size, layer_sizes)

    print('All test cases passed for layer sizes!\n')

def test_output(nn:NeuralNetwork, input_data:list, layer_sizes:list) -> None:
    """
    This function validates the shape of the output from the neural network's feed-forward pass.
    It checks if the output shape matches the expected size based on the input data and the size 
    of the output layer.
    Assumes:
    1. Neural network uses a feed-forward architecture.
    2. The nn.feed_forward(input_data) method returns a numpy array of shape (batch_size, output_layer_size).
    """

    output_layer_size = layer_sizes[-1]
    excpected_output_size = (input_data.shape[0], output_layer_size)
    output = nn.feed_forward(input_data)
    assert excpected_output_size == output.shape, \
        f'Output shape differs from excpected output shape: output shape = {output.shape}, excpected output shape = {excpected_output_size}'
    print('Output layer is as excpected!')

def perform_test_output():

    # Test case 1: 3 samples, 4 input features, [2, 3, 1] layer sizes with sigmoid activations
    X = np.random.randn(3, 4)
    nn = NeuralNetwork(4, [2, 3, 1], [sigmoid, sigmoid, sigmoid], loss='mse')
    test_output(nn, X, [2, 3, 1])

    # Test case 2: 5 samples, 4 input features, [3, 2, 1] with ReLU and sigmoid
    X = np.random.randn(5, 4)
    nn = NeuralNetwork(4, [3, 2, 1], [ReLU, sigmoid, sigmoid], loss='mse')
    test_output(nn, X, [3, 2, 1])

    # Test case 3: 10 samples, 6 input features, [5, 4, 2] with leakyReLU, softmax, sigmoid
    X = np.random.randn(10, 6)
    nn = NeuralNetwork(6, [5, 4, 2], [leakyReLU, sigmoid, sigmoid], loss='mse')
    test_output(nn, X, [5, 4, 2])

    # Test case 4: 2 samples, 3 input features, [3, 2] with sigmoid and softmax
    X = np.random.randn(2, 3)
    nn = NeuralNetwork(3, [3, 2], [sigmoid, sigmoid], loss='mse')
    test_output(nn, X, [3, 2])

    # Test case 5: 1 sample, 8 input features, [6, 5, 3, 2] with mixed activations
    X = np.random.randn(1, 8)
    nn = NeuralNetwork(8, [6, 5, 3, 2], [ReLU, leakyReLU, sigmoid, sigmoid], loss='mse')
    test_output(nn, X, [6, 5, 3, 2])

    # Test case 6: 50 samples, 20 input features, [10, 5, 1] with ReLU, softmax, sigmoid
    X = np.random.randn(50, 20)
    nn = NeuralNetwork(20, [10, 5, 1], [ReLU, sigmoid, sigmoid], loss='mse')
    test_output(nn, X, [10, 5, 1])

    # Test case 7: Minimal input, 1 sample, 2 input features, [1] layer size with sigmoid
    X = np.random.randn(1, 2)
    nn = NeuralNetwork(2, [1], [sigmoid], loss='mse')
    test_output(nn, X, [1])

    # Test case 8: 10 samples, 3 input features, 10 layers with random layer size with sigmoid
    X = np.random.randn(1, 2)
    layer_sizes = [np.random.randint(1, 100) for _ in range(50)]
    nn = NeuralNetwork(2, layer_sizes, 50*[sigmoid], loss='mse')
    test_output(nn, X, layer_sizes)

    print("All test cases passed for output shape!\n")

if __name__ == "__main__":
    perform_test_layers() 
    perform_test_output()
