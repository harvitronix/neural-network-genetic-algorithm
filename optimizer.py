"""
Class that holds a genetic algorithm for evolving a network.

Credit:
    A lot of those code was originally inspired by:
    http://lethain.com/genetic-algorithms-cool-name-damn-simple/
"""
from functools import reduce
from operator import add
import random
from network import Network

class Optimizer():
    """Class that implements genetic algorithm for MLP optimization."""

    def __init__(self, nn_param_choices, retain=0.6,
                 random_select=0.1, mutate_chance=0.1):
        """Create an optimizer with default options."""
        self.mutate_chance = mutate_chance
        self.random_select = random_select
        self.retain = retain
        self.nn_param_choices = nn_param_choices

    def create_population(self, count):
        """Create a population of random networks."""
        pop = []
        for _ in range(0, count):
            # Create a random network.
            network = Network(self.nn_param_choices)
            network.create_random()

            # Add the network to our population.
            pop.append(network)

        return pop

    @staticmethod
    def fitness(network):
        """Return the accuracy, which is our fitness function."""
        return network.accuracy

    def grade(self, pop):
        """Find average fitness for a population."""
        summed = reduce(add, (self.fitness(network) for network in pop))
        return summed / float((len(pop)))

    def breed(self, mother, father):
        """Make two children as parts of their parents."""
        children = []
        for _ in range(2):

            child = {}

            # Loop through the parameters and pick params for the kid.
            for key in self.nn_param_choices:
                child[param] = random.choice(
                    [mother.network[param], father.network[param]]
                )

            # Now create a network object.
            network = Network(self.nn_param_choices)
            network.create_set(child)

            children.append(network)

        return children

    def mutate(self, network):
        """Randomly mutate one part of the network."""

        # Choose a random key.
        mutation = random.choice(self.nn_param_choices)

        # Mutate one of the params.
        network.network[mutation] = random.choice(self.nn_param_choices[mutation])

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
