#!/usr/bin/env python3
"""
stride es los pasos, tipo si va de uno en uno
"""
import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """
    Performs a convolution on grayscale images with custom padding and stride.

    Parameters:
    - images: np.ndarray of shape (m, h, w)
    - kernel: np.ndarray of shape (kh, kw)
    - padding: 'same', 'valid', or (ph, pw)
    - stride: tuple (sh, sw)

    Returns:
    - np.ndarray of shape (m, out_h, out_w) with convolved images
    """
    m = len(images)
    h = len(images[0])
    w = len(images[0][0])
    kh = len(kernel)
    kw = len(kernel[0])
    ph, pw = padding
    sh, sw = stride

    if padding == 'same':
        # mantener el mismo tamaño
        padt = kh // 2
        padb = kh - 1 - padt
        padl = kw // 2
        padr = kw - 1 - padl

        padded_images = np.pad(
            images,
            pad_width=((0, 0), (padt, padb), (padl, padr)),
            mode='constant',
            constant_values=0
        )

        nuevoh = (h + padt + padb - kh) // sh + 1
        nuevow = (w + padl + padr - kw) // sw + 1
    # sin
    elif padding == 'valid':
        padded_images = images
        nuevoh = (h - kh) // sh + 1
        nuevow = (w - kw) // sw + 1

    else:
        # customizado
        ph, pw = padding
        padded_images = np.pad(
            images,
            pad_width=((0, 0), (ph, ph), (pw, pw)),
            mode='constant',
            constant_values=0
        )
        nuevoh = (h + 2 * ph - kh) // sh + 1
        nuevow = (w + 2 * pw - kw) // sw + 1

    nuevo = np.zeros((m, nuevoh, nuevow))

    for i in range(nuevoh):
        for j in range(nuevow):
            stai = i * sh
            staj = j * sw
            porcion = padded_images[:, stai:stai+kh, staj:staj+kw]
            nuevo[:, i, j] = np.sum(porcion * kernel, axis=(1, 2))

    return nuevo
