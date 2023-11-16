import numpy as np
import tensorflow as tf

from typing import List

class AutoEncoder(tf.keras.models.Model):

    def __init__(self, inputShape: List[int], latentShape: int, hiddenLayerShapes: List[int], hiddenLayerActivationFunction="relu", encoderOutputActivationFunction="sigmoid", decoderOutputActivationFunction="linear"):
        """
        Create a new AutoEncoder with defined layer shapes and activations.

        :param inputShape: A list of integers defining the data input shape. This is used for reshaping the data in the decoder output.
        :param latentShape: An int representing the shape of the latent layer (the layer of the encoder output).
        :param hiddenLayerShapes: A list of integers defining the layer shapes from input through to (and including) the final encoder layer.
            The first entry should be the first layer shape *after* the input. The last entry should be the shape of the final layer *before* the encoder output.
            This list is reversed and used for the decoder to give an hourglass network shape.
            If two separate lists are given, these are used first for the encoder layers, then the decoder.
        :param hiddenLayerActivationFunction: The activation function to use for the hidden layers (i.e. not the encoder output and not the decoder output).
        :param encoderOutputActivationFunction: The activation function to use for the encoder output.

        EXAMPLE: ----------------------------------------------------------------------------------

        (x_train, _), (x_test, _) = fashion_mnist.load_data()
        x_train = x_train.astype('float32') / 255.
        x_test = x_test.astype('float32') / 255.
        dataShape = x_test.shape[1:]

        autoencoder = AutoEncoder(inputShape=dataShape, latentShape=32, hiddenLayerShapes=[64, 32])
        autoencoder.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError())

        autoencoder.fit(x_train, x_train,
                epochs=10,
                shuffle=True,
                validation_data=(x_test, x_test))

        encoded_imgs = autoencoder.encoder(x_test).numpy()
        decoded_imgs = autoencoder.decoder(encoded_imgs).numpy()
        n = 10
        plt.figure(figsize=(20, 4))
        for i in range(n):
            # display original
            ax = plt.subplot(2, n, i + 1)
            plt.imshow(x_test[i])
            plt.title("original")
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # display reconstruction
            ax = plt.subplot(2, n, i + 1 + n)
            plt.imshow(decoded_imgs[i])
            plt.title("reconstructed")
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        plt.show()
        """

        super(AutoEncoder, self).__init__()
        self.inputShape = inputShape
        self.latentShape = latentShape
        self.hiddenLayerShapes = hiddenLayerShapes
        self.hiddenLayerActivationFunction = hiddenLayerActivationFunction
        self.encoderOutputActivationFunction = encoderOutputActivationFunction
        self.decoderOutputActivationFunction = decoderOutputActivationFunction

        encoderNetworkLayers = [tf.keras.layers.Flatten()]
        if type(self.hiddenLayerShapes[0]) is int:
            for shape in self.hiddenLayerShapes:
                encoderNetworkLayers.append(tf.keras.layers.Dense(shape, self.hiddenLayerActivationFunction))
            encoderNetworkLayers.append(tf.keras.layers.Dense(self.latentShape, self.encoderOutputActivationFunction, name="encoderOutput"))

            decoderNetworkLayers = []
            for shape in self.hiddenLayerShapes[::-1]:
                decoderNetworkLayers.append(tf.keras.layers.Dense(shape, hiddenLayerActivationFunction))
            decoderNetworkLayers.append(tf.keras.layers.Dense(tf.math.reduce_prod(self.inputShape), self.decoderOutputActivationFunction))
            decoderNetworkLayers.append(tf.keras.layers.Reshape(inputShape))

        else:
            for shape in self.hiddenLayerShapes[0]:
                encoderNetworkLayers.append(tf.keras.layers.Dense(shape, self.hiddenLayerActivationFunction))
            encoderNetworkLayers.append(tf.keras.layers.Dense(self.latentShape, self.encoderOutputActivationFunction, name="encoderOutput"))

            decoderNetworkLayers = []
            for shape in self.hiddenLayerShapes[1]:
                decoderNetworkLayers.append(tf.keras.layers.Dense(shape, hiddenLayerActivationFunction))
            decoderNetworkLayers.append(tf.keras.layers.Dense(tf.math.reduce_prod(self.inputShape), self.decoderOutputActivationFunction))
            decoderNetworkLayers.append(tf.keras.layers.Reshape(inputShape))

        self.encoderNetwork = tf.keras.Sequential(encoderNetworkLayers, name="encoder")
        self.decoderNetwork = tf.keras.Sequential(decoderNetworkLayers, name="decoder")
        self.build((1, np.prod(self.inputShape)))

    def call(self, X):
        encoded = self.encoderNetwork(X)
        decoded = self.decoderNetwork(encoded)
        return {
            "decoderOutput": decoded, 
            "encoderOutput": encoded
        }


class AutoEncoderSigmoidCustomLoss(tf.keras.losses.Loss):

    def call(self, _, y_pred):
        return -2*tf.reduce_mean(tf.math.square(y_pred)-y_pred, axis=-1)
        
class AutoEncoderTanhCustomLoss(tf.keras.losses.Loss):

    def call(self, _, y_pred):
        return -2*tf.reduce_mean(tf.math.square(y_pred)-1, axis=-1)