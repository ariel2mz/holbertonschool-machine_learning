#!/usr/bin/env python3
"""
sadsadas
dasdasds
"""
from tensorflow import keras as K
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Concatenate


def inception_block(A_prev, filters):
    """
    Builds an inception block as described in the 2014 GoogLeNet paper.

    Parameters:
    - A_prev: Output from the previous layer (tensor)
    - filters: Tuple/list of 6 filter sizes: (F1, F3R, F3, F5R, F5, FPP)

    Returns:
    - The concatenated output of the inception block
    """
    F1, F3R, F3, F5R, F5, FPP = filters

    con1x1A = K.layers.Conv2D(filters=F1,
                              kernel_size=(1, 1),
                              padding='same',
                              activation='relu')(A_prev)

    # Ruta B con1x1B -> con3x3B
    con1x1B = K.layers.Conv2D(filters=F3R,
                              kernel_size=(1, 1),
                              padding='same',
                              activation='relu')(A_prev)
    con3x3B = K.layers.Conv2D(filters=F3,
                              kernel_size=(3, 3),
                              padding='same',
                              activation='relu')(con1x1B)

    # Ruta C con1x1 -> con5x5
    con1x1C = K.layers.Conv2D(filters=F5R, kernel_size=(1, 1),
                              padding='same', activation='relu')(A_prev)
    con5x5C = K.layers.Conv2D(filters=F5, kernel_size=(5, 5),
                              padding='same', activation='relu')(con1x1C)

    # Ruta D maxpooling3x3 -> conv1x1
    maxpoolD = K.layers.MaxPooling2D(pool_size=(3, 3), strides=(1, 1),
                                     padding='same')(A_prev)
    con1x1D = K.layers.Conv2D(filters=FPP, kernel_size=(1, 1),
                              padding='same', activation='relu')(maxpoolD)

    output = K.layers.Concatenate(axis=-1)([con1x1A,
                                            con3x3B,
                                            con5x5C,
                                            con1x1D])

    return output
