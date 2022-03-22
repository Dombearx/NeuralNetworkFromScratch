import random
from collections.abc import Callable
import math

import numpy as np
import matplotlib.pyplot as plt


class Neuron:

    def __init__(self, previous_layer_size: int, activation_function: Callable[[float], float],
                 random_initialization: bool = True) -> None:

        if random_initialization:
            self.weights = np.random.rand(previous_layer_size)
            self.bias = random.random()
        else:
            self.weights = np.ones(previous_layer_size)
            self.bias = 1

        self.activation_function = activation_function

    def predict(self, values: np.ndarray) -> float:
        return self.activation_function(np.sum(values * self.weights) + self.bias)

    def learn(self):
        raise Exception('Not implemented yet!')


class Layer:
    def __init__(self, previous_layer_size: int, activation_function: Callable[[float], float],
                 number_of_neurons: int = 10, random_initialization: bool = True):
        self.neurons = [Neuron(previous_layer_size, activation_function, random_initialization)
                        for _ in range(number_of_neurons)]

    def predict(self, values: np.ndarray) -> np.ndarray:
        return np.array([neuron.predict(value) for neuron, value in zip(self.neurons, values)])

    def learn(self):
        raise Exception('Not implemented yet!')


class Network:
    def __init__(self, layer_sizes: tuple, activation_function: Callable[[float], float],
                 random_initialization: bool = True):
        self.layers = [Layer(previous_layer_size, activation_function, number_of_neurons, random_initialization)
                       for previous_layer_size, number_of_neurons in zip(layer_sizes[:-1], layer_sizes[1:])]

    def predict(self, values: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            values = layer.predict(values)

        return values

    def learn(self, values: np.ndarray, true_label: int) -> None:
        pass


def target_function(x: int) -> int:
    return x ** 2 % 10


def main():
    print('Simple neural network project')

    network = Network((16, 12, 12, 10), lambda v: 1 / (1 + math.exp(-v)))

    input_val = np.array(list(map(int, '{0:b}'.format(20).zfill(16))))

    network_predicted = network.predict(input_val)


    x = np.arange(-100, 100, 1)

    y = np.array(list(map(target_function, x)))

    fig = plt.figure()
    plt.plot(x, y, 'r')
    plt.show()


if __name__ == '__main__':
    main()
