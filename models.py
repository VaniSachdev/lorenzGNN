import tensorflow as tf
from tensorflow.keras import Model


class NaiveZero(Model):
    """ This model predicts an output of 0 for any input. """

    def __init__(self):
        super().__init__()

    def call(self, inputs):
        """ 
            input shape: (batches, K, 2 x num_time_steps)
            output shape: (batches, K, 1)
        """
        if len(inputs) == 2:
            x, a = inputs
        else:
            x, a, _ = inputs  # So that the model can be used with DisjointLoader

        # x has shape (batches, K, 2 x features)
        b = len(x)
        K = len(x[0])
        zeros = tf.zeros(shape=(b, K, 1))
        return zeros


class NaiveConstant(Model):
    """ This model predicts that the output is equal to the last X1 value in 
        the input window.
    """

    def __init__(self):
        super().__init__()

    def call(self, inputs):
        """ 
            input shape: (batches, K, 2 x num_time_steps)
            output shape: (batches, K, 1)
        """
        if len(inputs) == 2:
            x, a = inputs
        else:
            x, a, _ = inputs  # So that the model can be used with DisjointLoader

        # x has shape (batches, K, 2 x features)
        # we want to get the last value of the X1 variable
        return x[:, :, len(x[0][0]) // 2 - 1, tf.newaxis]
