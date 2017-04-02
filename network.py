"""Class that represents the network to be evolved."""
import random
from train import train_and_score

class Network():
    """Represent a network and let us operate on it.

    Currently only works for an MLP.
    """

    def __init__(self, neuron_choices, max_layers=4):
        """Initialize our network.

        Args:
            neuron_choices (list): List of available layer widths
                For example, [24, 32, 512, 1024]
            max_layers (int): Maximum depth of the network

        """
        self.accuracy = 0.
        self.neuron_choices = neuron_choices
        self.max_layers = max_layers
        self.network = []  # (list): represents MLP network

    def create_random_network(self):
        """Create a random network."""

        # Start with random number of layers.
        nb_layers = random.randint(1, self.max_layers)

        # For each layer, get random number of neurons.
        for _ in range(nb_layers):
            nb_neurons = random.choice(self.neuron_choices)
            self.network.append(nb_neurons)

    def create_set_network(self, network):
        """Set network properties.

        Args:
            network (list): List of neurons per layer.

        """
        self.network = network

    def train_network(self):
        """Train the network and record the accuracy."""
        if self.accuracy == 0.:
            self.accuracy = train_and_score(self.network)

    def print_network(self):
        """Print out a network."""
        print(self.network)
        print("Network accuracy: %.2f%%" % (self.accuracy * 100))
