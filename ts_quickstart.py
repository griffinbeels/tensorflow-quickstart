# Tensorflow is the main library for creating our NN
import tensorflow as tf

def load_data():
    """ Loads from the MNIST data set, for both training
    and testing sets.

    input:
        none

    output:
        x_train: data points for training
        y_train: labels for training
        x_test: data points for testing
        y_test: labels for testing
    """
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    return x_train, y_train, x_test, y_test

def create_model():
    """ Returns the sequential model by stacking layers.

    output:
        model: neural net w/ a few layers
    """
    return tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10)
    ])

def train_and_fit(model, x_train, y_train):
    """ Compiles the model using the 'adam' optimizer,
    and fits over 5 epochs.  Prints accuracy and loss
    at each epoch.

    input: 
        model: the model representing our neural net
        x_train: the training data
        y_train: the training labels
    
    output: 
        none
    """
    predictions = model(x_train[:1]).numpy()

    # softmax converts logits to probability 
    tf.nn.softmax(predictions).numpy()

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    loss_fn(y_train[:1], predictions).numpy()

    model.compile(optimizer='adam',
                loss=loss_fn,
                metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=5)

def evaluate_model(model, x_test, y_test):
    """ Evaluates the given model using the provided
    testing data and labels.  Prints probability!

    input:
        model: the model after being trained
        x_test: testing data split from original set
        y_test: testing labels "" 

    output:
        none
    """
    model.evaluate(x_test,  y_test, verbose=2)
    probability_model = tf.keras.Sequential([
    model,
    tf.keras.layers.Softmax()
    ])
    probability_model(x_test[:5])

def main():
    """ Sets up our model, trains, and evaluates it. """
    x_train, y_train, x_test, y_test = load_data()
    model = create_model()
    train_and_fit(model, x_train, y_train)
    evaluate_model(model, x_test, y_test)
    
main()