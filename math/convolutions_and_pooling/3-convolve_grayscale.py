#!/usr/bin/env python3
"""
le pedi a deepseek q me lo haga mejor
porque mi codigo estaba todo bien
menos el valid que me daba un output distinto
y no encontraba que era el error
(mi codigo era bastante distinto a este
en cada if hacia cada output con 0s pero
las ias no lograban decirme que estaba mal
asiq lo hice de cero)
"""
import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """
    Performs 2D convolution operation on multiple grayscale images

    Process overview:
    1. Handles different padding schemes
    2. Applies sliding window operation
    3. Computes element-wise multiplication and summation

    Args:
        images: Input array (num_images, height, width)
        kernel: Convolution filter (k_height, k_width)
        padding: Padding strategy ('same'/'valid') or custom (ph, pw)
        stride: Step sizes for convolution (sh, sw)

    Returns:
        Convolved output (num_images, out_height, out_width)
    """
    # Extract dimensional parameters
    m, h, w = images.shape
    kh, kw = kernel.shape
    sh, sw = stride

    # Process padding configuration
    if padding == 'same':
        ph = ((h - 1) * sh + kh - h + 1) // 2
        pw = ((w - 1) * sw + kw - w + 1) // 2
    elif padding == 'valid':
        ph = pw = 0
    else:
        ph, pw = padding

    # Initialize padded array
    padd = np.zeros((m, h + 2*ph, w + 2*pw))
    padd[:, ph:ph+h, pw:pw+w] = images

    # Calculate output dimensions
    nuevoh = (h + 2*ph - kh) // sh + 1
    nuevow = (w + 2*pw - kw) // sw + 1

    # Prepare output tensor
    nuevo = np.zeros((m, nuevoh, nuevow))

    # Main convolution operation
    for i in range(nuevoh):
        hstart = i * sh
        for j in range(nuevow):
            wstart = j * sw

            # Extract current receptive field
            region = padd[:,
                        hstart:hstart+kh, 
                        wstart:wstart+kw]

            # Compute convolution result
            nuevo[:, i, j] = np.sum(region * kernel, axis=(1, 2))

    return nuevo
