#!/usr/bin/env python3
"""sadasdsadsa"""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    dasdsadsa asdsadsa
    """
    # ----- Encoder -----
    ilayer = keras.Input(shape=(input_dims,))
    x = ilayer
    for units in hidden_layers:
        x = keras.layers.Dense(units, activation="relu")(x)
    lat = keras.layers.Dense(latent_dims, activation="relu")(x)

    encoder = keras.Model(inputs=ilayer, outputs=lat, name="encoder")

    # ----- Decoder -----
    latinp = keras.Input(shape=(latent_dims,))
    x = latinp
    for units in reversed(hidden_layers):
        x = keras.layers.Dense(units, activation="relu")(x)
    output_layer = keras.layers.Dense(input_dims, activation="sigmoid")(x)

    decoder = keras.Model(inputs=latinp, outputs=output_layer, name="decoder")

    # ----- Autoencoder -----
    auto_outputs = decoder(encoder(ilayer))
    auto = keras.Model(inputs=ilayer, outputs=auto_outputs, name="autoencoder")

    auto.compile(optimizer="adam", loss="binary_crossentropy")

    return encoder, decoder, auto
