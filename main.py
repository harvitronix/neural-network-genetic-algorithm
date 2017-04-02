"""The main driver for evolving a network. Start here."""
from generate import generate

def main():
    """Evolve a network."""
    generations = 5  # Number of times to evole the population.
    population = 20  # Number of networks in each generation.

    neuron_choices = [64, 128, 256, 512, 768, 1024]
    max_layers = 4

    print("***Evolving %d generations with population %d***" %
          (generations, population))

    generate(generations, population, neuron_choices, max_layers)

if __name__ == '__main__':
    main()

