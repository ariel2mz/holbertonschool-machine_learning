#!/usr/bin/env python3
"""sadasdsadsa"""
import tensorflow.keras as keras
import tensorflow as tf


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    sdasdadas
    """

    inputs = keras.Input(shape=(input_dims,))
    x = inputs
    for units in hidden_layers:
        x = keras.layers.Dense(units, activation="relu")(x)

    zmean = keras.layers.Dense(latent_dims, activation=None)(x)
    zlogvar = keras.layers.Dense(latent_dims, activation=None)(x)


    def sampling(args):
        """
        sadasdsadas
        """
        zmean, zlogvar = args
        epsilon = tf.keras.backend.random_normal(shape=tf.shape(zmean))
        return zmean + tf.exp(0.5 * zlogvar) * epsilon

    z = keras.layers.Lambda(sampling)([zmean, zlogvar])

    encoder = keras.Model(inputs, [z, zmean, zlogvar], name="encoder")

    latinp = keras.Input(shape=(latent_dims,))
    x = latinp
    for units in reversed(hidden_layers):
        x = keras.layers.Dense(units, activation="relu")(x)
    outputs = keras.layers.Dense(input_dims, activation="sigmoid")(x)

    decoder = keras.Model(latinp, outputs, name="decoder")

    z, zmean, zlogvar = encoder(inputs)
    reconstructed = decoder(z)

    auto = keras.Model(inputs, reconstructed, name="variational_autoencoder")

    reconstruction_loss = keras.losses.binary_crossentropy(inputs, reconstructed)
    reconstruction_loss = tf.reduce_sum(reconstruction_loss, axis=1)
    kloss = -0.5 * tf.reduce_sum(1 + zlogvar - tf.square(zmean) - tf.exp(zlogvar), axis=1)
    vaeloss = tf.reduce_mean(reconstruction_loss + kloss)
    auto.add_loss(vaeloss)

    auto.compile(optimizer="adam")

    return encoder, decoder, auto
