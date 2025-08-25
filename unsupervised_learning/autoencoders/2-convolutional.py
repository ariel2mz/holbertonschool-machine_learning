#!/usr/bin/env python3
"""sadasdsadsa"""
import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """
    asdsadsadsadsaa
    """

    input_layer = keras.Input(shape=input_dims)
    x = input_layer

    for f in filters:
        x = keras.layers.Conv2D(filters=f, kernel_size=(3, 3),
                                padding="same", activation="relu")(x)
        x = keras.layers.MaxPooling2D(pool_size=(2, 2), padding="same")(x)

    encoder_output = x
    encoder = keras.Model(inputs=input_layer, outputs=encoder_output, name="encoder")

    latinp = keras.Input(shape=latent_dims)
    x = latinp

    for f in reversed(filters[:-1]):
        x = keras.layers.Conv2D(filters=f, kernel_size=(3, 3),
                                padding="same", activation="relu")(x)
        x = keras.layers.UpSampling2D(size=(2, 2))(x)

    x = keras.layers.Conv2D(filters=filters[0], kernel_size=(3, 3),
                            padding="valid", activation="relu")(x)

    output_layer = keras.layers.Conv2D(filters=input_dims[-1], kernel_size=(3, 3),
                                       padding="same", activation="sigmoid")(x)

    decoder = keras.Model(inputs=latinp, outputs=output_layer, name="decoder")

    autout = decoder(encoder(input_layer))
    auto = keras.Model(inputs=input_layer, outputs=autout, name="autoencoder")

    auto.compile(optimizer="adam", loss="binary_crossentropy")

    return encoder, decoder, auto
