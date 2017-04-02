"""
Utility used by the Network class to actually train.

Based on:
    https://github.com/fchollet/keras/blob/master/examples/mnist_mlp.py

TODO:
* Use early stopping instead of a set epoch
* Parameterize everything
"""
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping

# Set defaults.
nb_classes = 10
batch_size = 128
input_shape = (784,)

# Get the data.
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# convert class vectors to binary class matrices
y_train = to_categorical(y_train, nb_classes)
y_test = to_categorical(y_test, nb_classes)

# Helper: Early stopping.
early_stopper = EarlyStopping(patience=5)

def compile_model(network):
    """Compile a sequential model.

    Args:
        network (list): a list of layers.

    Returns:
        a compiled network.

    """
    model = Sequential()

    # Add each layer.
    for i, neurons in enumerate(network):

        # Need input shape for first layer.
        if i == 0:
            model.add(Dense(neurons, input_shape=input_shape))
        else:
            model.add(Dense(neurons))

        model.add(Dropout(0.2))  # hard-coded dropout

    # Output layer.
    model.add(Dense(nb_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam',
                  metrics=['accuracy'])

    return model

def train_and_score(network):
    """Train the model, return test loss.

    Args:
        network (list): a list of layers.

    """
    model = compile_model(network)

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=10000,  # using early stopping, so no real limit
              verbose=0,
              validation_data=(x_test, y_test),
              callbacks=[early_stopper])

    score = model.evaluate(x_test, y_test, verbose=0)

    #print(network)
    #print("Test accuracy: %.2f%%" % (score[1] * 100))

    return score[1]  # 1 is accuracy. 0 is loss.
