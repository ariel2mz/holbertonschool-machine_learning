#!/usr/bin/env python3
"""sadasdsadsa"""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims, lambtha):
    """
    asdfghjkh
    """

    inplayer = keras.Input(shape=(input_dims,))
    x = inplayer
    for units in hidden_layers:
        x = keras.layers.Dense(units, activation="relu")(x)
    lat = keras.layers.Dense(latent_dims, activation="relu",
                             activity_regularizer=keras.regularizers.l1(
                                lambtha))(x)

    encoder = keras.Model(inputs=inplayer, outputs=lat, name="encoder")

    latinp = keras.Input(shape=(latent_dims,))
    x = latinp
    for units in reversed(hidden_layers):
        x = keras.layers.Dense(units, activation="relu")(x)
    outlayer = keras.layers.Dense(input_dims, activation="sigmoid")(x)

    decoder = keras.Model(inputs=latinp, outputs=outlayer, name="decoder")

    autout = decoder(encoder(inplayer))
    auto = keras.Model(inputs=inplayer,
                       outputs=autout, name="sparse_autoencoder")

    auto.compile(optimizer="adam", loss="binary_crossentropy")

    return encoder, decoder, auto
