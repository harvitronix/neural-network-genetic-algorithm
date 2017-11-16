# Evolve a neural network with a genetic algorithm

This is an example of how we can use a genetic algorithm in an attempt to find the optimal network parameters for classification tasks.

It's currently limited to only MLPs (ie. fully connected networks) and uses the Keras library to build, train and validate.

On the easy MNIST dataset, we are able to quickly find a network that reaches > 98% accuracy. On the more challenging CIFAR10 dataset, we get to 56% after 10 generations (with population 20).

For more, see this blog post: 
https://medium.com/@harvitronix/lets-evolve-a-neural-network-with-a-genetic-algorithm-code-included-8809bece164

For a more robust implementation that you can use in your projects, take a look at [Jan Liphardt's implementation, DeepEvolve](https://github.com/jliphard/DeepEvolve).

## To run

To run the brute force algorithm:

```python3 brute.py```

To run the genetic algorithm:

```python3 main.py```

You can set your network parameter choices by editing each of those files first. You can also choose whether to use the MNIST or CIFAR10 datasets. Simply set `dataset` to either `mnist` or `cifar10`.

## Credits

The genetic algorithm code is based on the code from this excellent blog post: https://lethain.com/genetic-algorithms-cool-name-damn-simple/

## Contributing

Have an optimization, idea, suggestion, bug report? Pull requests greatly appreciated!

## License

MIT
