import random


class Neuron:

    def __init__(self, random_initialization: bool = True) -> None:
        if random_initialization:
            self.weight = random.random()
            self.bias = random.random()
        else:
            self.weight = 1
            self.bias = 1

    def predict(self, value: float) -> float:
        return value * self.weight + self.bias

    def learn(self):
        raise Exception('Not implemented yet!')


class Layer:
    def __init__(self, number_of_neurons: int = 100, random_initialization: bool = True):
        self.neurons = [Neuron(random_initialization) for _ in range(number_of_neurons)]

    def predict(self, values: []) -> []:
        return [neuron.predict(value) for neuron, value in zip(self.neurons, values)]

    def learn(self):
        raise Exception('Not implemented yet!')


def function(x: float) -> float:
    return x + 6


def main():
    print('Simple neural network project')


if __name__ == '__main__':
    main()
