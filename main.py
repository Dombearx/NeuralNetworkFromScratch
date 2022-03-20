import random


class Neuron:

    def __init__(self, random_initialization: bool = True) -> None:
        if random_initialization:
            self.weight = random.random()
            self.bias = random.random()
        else:
            self.weight = 1
            self.bias = 1

    def forward(self, value: float) -> float:
        return value * self.weight + self.bias

    def backward(self):
        raise Exception('Not implemented yet!')


def main():
    print('Simple neural network project')


if __name__ == '__main__':
    main()
