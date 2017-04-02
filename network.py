"""Class that represents the network to be evolved."""
import random
from train import train_and_score

class Network():
    """Represent a network and let us operate on it.

    Currently only works for an MLP.
    """

    def __init__(self, nn_params):
        """Initialize our network.

        Args:
            nn_params (dict): Parameters for the network.
                Should include:
                neuron_choices (list): example [64, 128, 256]
                max_layers (int): ie 4 = 4 layers plus output
                activations (list): ie ['relu', 'elu']
                optimizers (list) ie ['rmsprop', 'adam']
        """
        self.accuracy = 0.
        self.nn_params = nn_params
        self.network = {}  # (dic): represents MLP network parameters

    def create_random(self):
        """Create a random network."""

        # Get number of layers.
        nb_layers = random.randint(1, self.nn_params['max_layers'])

        # Get a random number of neurons for the layers.
        nb_neurons = random.choice(self.nn_params['neuron_choices'])

        # Choose an activation.
        activation = random.choice(self.nn_params['activations'])

        # Choose an optimizer.
        optimizer = random.choice(self.nn_params['optimizers'])

        # Now build our network.
        self.network = {
            'nb_layers': nb_layers,
            'nb_neurons': nb_neurons,
            'activation': activation,
            'optimizer': optimizer,
        }

    def create_set(self, network):
        """Set network properties.

        Args:
            network (list): List of neurons per layer.

        """
        self.network = network

    def train(self):
        """Train the network and record the accuracy."""
        if self.accuracy == 0.:
            self.accuracy = train_and_score(self.network)

    def print(self):
        """Print out a network."""
        print(self.network)
        print("Network accuracy: %.2f%%" % (self.accuracy * 100))
