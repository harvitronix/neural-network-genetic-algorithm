"""
Class that holds a genetic algorithm for evolving a network.

Credit:
    A lot of those code was originally inspired by:
    http://lethain.com/genetic-algorithms-cool-name-damn-simple/
"""
from functools import reduce
from operator import add
import random
import sys
from network import Network

class Optimizer():
    """Class that implements genetic algorithm for MLP optimization."""

    def __init__(self, neuron_choices, max_layers, retain=0.6, 
                 random_select=0.05, mutate_chance=0.005):
        """Create an optimizer with default options."""
        self.mutate_chance = mutate_chance
        self.random_select = random_select
        self.retain = retain
        self.neuron_choices = neuron_choices
        self.max_layers = max_layers

    def create_population(self, count):
        """Create a population of random networks."""
        pop = []
        for _ in range(0, count):
            network = Network(self.neuron_choices, self.max_layers)
            network.create_random_network()
            pop.append(network)
        return pop

    def fitness(self, network):
        """Return the accuracy, which is our fitness function."""
        return network.accuracy

    def grade(self, pop):
        """Find average fitness for a population."""
        summed = reduce(add, (self.fitness(network) for network in pop))
        return summed / float((len(pop)))

    def breed(self, mother, father):
        """Make two children as parts of their parents."""
        mother_depth = len(mother.network)
        father_depth = len(father.network)

        # We'll create two children.
        children = []
        for _ in range(2):

            # Randomly choose one of the depths.
            child_depth = random.choice([mother_depth, father_depth])

            # Now breed the widths.
            child = []
            for i in range(child_depth):
                # Randomly get mother or father for this layer.
                if random.random() > 0.5:
                    # Use the min of i and the last layer.
                    child.append(mother.network[min(i, mother_depth-1)])
                else:
                    child.append(father.network[min(i, father_depth-1)])

            # Now create a network object.
            network = Network(self.neuron_choices, self.max_layers)
            network.create_set_network(child)

            children.append(network)

        return children

    def mutate(self, network):
        """Randomly mutate one part of the network."""
        random_layer = random.randint(0, len(network.network) - 1)

        # Update one layer with a random neuron count.
        network.network[random_layer] = random.choice(self.neuron_choices)

        return network

    def evolve(self, pop):
        """The main algorithm function. Evolve a population of networks."""
        graded = [(self.fitness(network), network) for network in pop]
        graded = [x[1] for x in sorted(graded, key=lambda x: x[0], reverse=True)]
        retain_length = int(len(graded)*self.retain)
        parents = graded[:retain_length]

        # randomly add other individuals to promote genetic diversity
        for individual in graded[retain_length:]:
            if self.random_select > random.random():
                parents.append(individual)

        # mutate some individuals
        for individual in parents:
            if self.mutate_chance > random.random():
                individual = self.mutate(individual)

        # crossover parents to create children
        parents_length = len(parents)
        desired_length = len(pop) - parents_length
        children = []
        while len(children) < desired_length:
            male = random.randint(0, parents_length-1)
            female = random.randint(0, parents_length-1)
            if male != female:
                male = parents[male]
                female = parents[female]
                babies = self.breed(male, female)
                for baby in babies:
                    # Don't grow larger than desired length.
                    if len(children) < desired_length:
                        children.append(baby)

        parents.extend(children)

        return parents
