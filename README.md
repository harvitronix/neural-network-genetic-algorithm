# Evolve a neural network with a genetic algorithm

This is an example of how we can use a genetic algorithm in an attempt to find the optimal network parameters for classification tasks.

It's currently limited to only MLPs (ie. fully connected networks) and uses the Keras library to build, train and validate.

On the easy MNIST dataset, we are able to quickly find a network that reaches > 90% accuracy. On the slightly more challenging CIFAR10 dataset, we get to X% after Y generations (with population 20).

For more, see this blog post: 
