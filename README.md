# Evolve a neural network with a genetic algorithm

This is an example of how we can use a genetic algorithm in an attempt to find the optimal network parameters for classification tasks.

It's currently limited to only MLPs (ie. fully connected networks) and uses the Keras library to build, train and validate.

On the easy MNIST dataset, we are able to quickly find a network that reaches > 98% accuracy. On the more challenging CIFAR10 dataset, we get to 56% after 10 generations (with population 20).

For more, see this blog post: 
https://medium.com/@harvitronix/lets-evolve-a-neural-network-with-a-genetic-algorithm-code-included-8809bece164
