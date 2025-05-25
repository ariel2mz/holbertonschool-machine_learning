#!/usr/bin/env python3
"""
como el anterior bro pero hasta mas facil incluso
"""
import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """
    Performs a convolution on grayscale images with custom padding.

    Parameters:
    - images: np.ndarray of shape (m, h, w)
    - kernel: np.ndarray of shape (kh, kw)
    - padding: tuple of (ph, pw)

    Returns:
    - np.ndarray containing the convolved images
    """
    m = len(images)
    h = len(images[0])
    w = len(images[0][0])
    kh = len(kernel)
    kw = len(kernel[0])
    ph, pw = padding

    padded_images = np.pad(
        images,
        pad_width=((0, 0), (ph, ph), (pw, pw)),
        mode='constant',
        constant_values=0
    )

    nuevoh = h + 2 * ph - kh + 1
    nuevow = w + 2 * pw - kw + 1
    nuevo = np.zeros((m, nuevoh, nuevow))

    for i in range(nuevoh):
        for j in range(nuevow):
            porcion = padded_images[:, i:i+kh, j:j+kw]
            pmulti = porcion * kernel
            psumad = np.sum(pmulti, axis=(1, 2))
            nuevo[:, i, j] = psumad

    return nuevo
