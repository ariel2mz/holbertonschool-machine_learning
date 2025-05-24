#!/usr/bin/env python3
"""
In a "same" convolution, the output image should have the same
height and width as the input image.
Since the convolution kernel "eats" into the image's edges,
we need to add padding (zero values)
around the borders of the input image to compensate.
"""
import numpy as np


def convolve_grayscale_same(images, kernel):
    """
    Performs a same convolution on grayscale images.

    Parameters:
    - images: np.ndarray with shape (m, h, w)
    - kernel: np.ndarray with shape (kh, kw)

    Returns:
    - np.ndarray containing the convolved images
    """
    m = len(images)
    h = len(images[0])
    w = len(images[0][0])
    kh = len(kernel)
    kw = len(kernel[0])

    padh = kh // 2
    padw = kw // 2

    padt = padh
    padb = kh - 1 - padh
    padl = padw
    padr = kw - 1 - padw

    padded_images = np.pad(
        images,
        pad_width=((0, 0), (padt, padb), (padl, padr)),
        mode='constant',
        constant_values=0
    )

    nuevo = np.zeros((m, h, w))

    for i in range(h):
        for j in range(w):
            porcion = padded_images[:, i:i+kh, j:j+kw]
            pmulti = porcion * kernel
            psumad = np.sum(pmulti, axis=(1, 2))
            nuevo[:, i, j] = psumad

    return nuevo
