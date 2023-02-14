import tensorflow as tf
from tensorflow.keras import Model
from spektral.layers.convolutional import gcn_conv


class GCN3(Model):
    """
        Args:
        - `n_labels`: number of channels in output;
        - `channels_0`: number of channels in first GCNConv layer;
        - `channels_1`: number of channels in second GCNConv layer;
        - `activation`: activation of the first GCNConv layer;
        - `output_activation`: activation of the second GCNConv layer;
        - `use_bias`: whether to add a learnable bias to the two GCNConv layers;
        - `dropout_rate`: `rate` used in `Dropout` layers;
        - `l2_reg`: l2 regularization strength;
        - `**kwargs`: passed to `Model.__init__`.
    """

    def __init__(
        self,
        n_labels=1,
        channels_0=2048,
        channels_1=32,
        activation="relu",
        output_activation=None,
        use_bias=False,
        dropout_rate=0.5,
        l2_reg=2.5e-4,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.n_labels = n_labels
        self.channels_0 = channels_0
        self.channels_1 = channels_1
        self.activation = activation
        self.output_activation = output_activation
        self.use_bias = use_bias
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
        reg = tf.keras.regularizers.l2(l2_reg)
        self._d0 = tf.keras.layers.Dropout(dropout_rate)
        self._gcn0 = gcn_conv.GCNConv(channels_0,
                                      activation=activation,
                                      kernel_regularizer=reg,
                                      use_bias=use_bias)
        self._d1 = tf.keras.layers.Dropout(dropout_rate)
        self._gcn1 = gcn_conv.GCNConv(channels_1,
                                      activation=activation,
                                      kernel_regularizer=reg,
                                      use_bias=use_bias)
        self._d2 = tf.keras.layers.Dropout(dropout_rate)
        self._gcn2 = gcn_conv.GCNConv(n_labels,
                                      activation=output_activation,
                                      use_bias=use_bias)

    def get_config(self):
        return dict(
            n_labels=self.n_labels,
            channels_0=self.channels_0,
            channels_1=self.channels_1,
            activation=self.activation,
            output_activation=self.output_activation,
            use_bias=self.use_bias,
            dropout_rate=self.dropout_rate,
            l2_reg=self.l2_reg,
        )

    def call(self, inputs):
        if len(inputs) == 2:
            x, a = inputs
        else:
            x, a, _ = inputs  # So that the model can be used with DisjointLoader

        x = self._d0(x)
        x = self._gcn0([x, a])
        x = self._d1(x)
        return self._gcn1([x, a])


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
