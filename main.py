import random

import numpy as np
import matplotlib.pyplot as plt


class Neuron:

    def __init__(self, previous_layer_size: int, random_initialization: bool = True) -> None:
        if random_initialization:
            self.weights = np.random.rand(previous_layer_size)
            self.biases = np.random.rand(previous_layer_size)
        else:
            self.weights = np.ones(previous_layer_size)
            self.biases = np.ones(previous_layer_size)

    def predict(self, values: np.ndarray) -> np.ndarray:
        return (values * self.weights) + self.biases

    def learn(self):
        raise Exception('Not implemented yet!')


class Layer:
    def __init__(self, previous_layer_size: int, number_of_neurons: int = 10, random_initialization: bool = True):
        self.neurons = [Neuron(previous_layer_size, random_initialization) for _ in range(number_of_neurons)]

    def predict(self, values: np.ndarray) -> np.ndarray:
        return np.array([neuron.predict(value) for neuron, value in zip(self.neurons, values)])

    def learn(self):
        raise Exception('Not implemented yet!')


def target_function(x: int) -> int:
    return x**2 % 10


def main():
    print('Simple neural network project')

    x = np.arange(-100, 100, 1)

    y = np.array(list(map(target_function, x)))

    fig = plt.figure()
    plt.plot(x, y, 'r')
    plt.show()


if __name__ == '__main__':
    main()
