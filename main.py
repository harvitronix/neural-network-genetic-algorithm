"""Entry point to evolving the neural network. Start here."""
from optimizer import Optimizer
from tqdm import tqdm

def train_networks(networks):
    """Train each network.

    Args:
        networks (list): Current population of networks.

    """
    pbar = tqdm(total=len(networks))
    for network in networks:
        network.train()
        network.print()
        pbar.update(1)
    pbar.close()

def get_average_accuracy(networks):
    """Get the average accuracy for a group of networks.

    Args:
        networks (list): List of networks.

    Returns:
        float: The average accuracy of a population of networks.

    """
    total_accuracy = 0
    for network in networks:
        total_accuracy += network.accuracy

    return total_accuracy / len(networks)

def generate(generations, population, nn_param_choices):
    """Generate a network with the genetic algorithm.

    Args:
        generations (int): Number of times to evole the population.
        population (int): Number of networks in each generation.

    """
    optimizer = Optimizer(nn_param_choices)
    networks = optimizer.create_population(population)

    # Evolve the generation.
    for i in range(generations):
        print("***Doing generation %d of %d***" %
              (i + 1, generations))

        # Train and get accuracy for networks.
        train_networks(networks)

        # Get the average accuracy for this generation.
        average_accuracy = get_average_accuracy(networks)

        # Print out the average accuracy each generation.
        print("Generation average: %.2f%%" % (average_accuracy * 100))
        print('-'*80)

        # Evolve, except on the last iteration.
        if i != generations - 1:
            # Do the evolution.
            networks = optimizer.evolve(networks)

    # Sort our final population.
    networks = sorted(networks, key=lambda x: x.accuracy, reverse=True)

    # Print out the top 5 networks.
    print_networks(networks[:5])

def print_networks(networks):
    """Print a list of networks.

    Args:
        networks (list of lists): A list of networks.

    """
    print('-'*80)
    for network in networks:
        network.print()

def main():
    """Evolve a network."""
    generations = 5  # Number of times to evole the population.
    population = 20  # Number of networks in each generation.

    nn_param_choices = {
        'nb_neurons': [64, 128, 256, 512, 768, 1024],
        'nb_layers': [1, 2, 3, 4],
        'activation': ['relu', 'elu', 'tanh', 'sigmoid'],
        'optimizer': ['rmsprop', 'adam', 'sgd', 'adagrad',
                       'adadelta', 'adamax', 'nadam'],
    }

    print("***Evolving %d generations with population %d***" %
          (generations, population))

    generate(generations, population, nn_param_choices)

if __name__ == '__main__':
    main()
