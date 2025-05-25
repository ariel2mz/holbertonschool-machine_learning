#!/usr/bin/env python3
"""
m ayudo deepseek porque mi codigo no
andaba el 'valid' no se por que literal
no tengo ni idea
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
    m, h, w = images.shape
    kh, kw = kernel.shape
    sh, sw = stride

    if padding == 'same':
        padt = (kh - 1) // 2
        padb = kh - 1 - padt
        padl = (kw - 1) // 2
        padr = kw - 1 - padl
        padded_images = np.pad(
            images,
            pad_width=((0, 0), (padt, padb), (padl, padr)),
            mode='constant',
            constant_values=0
        )
    elif padding == 'valid':
        # No padding
        padded_images = images
    else:
        # Custom padding
        ph, pw = padding
        padded_images = np.pad(
            images,
            pad_width=((0, 0), (ph, ph), (pw, pw)),
            mode='constant',
            constant_values=0
        )

    padded_h = padded_images.shape[1]
    padded_w = padded_images.shape[2]
    out_h = (padded_h - kh) // sh + 1
    out_w = (padded_w - kw) // sw + 1

    output = np.zeros((m, out_h, out_w))

    for i in range(out_h):
        for j in range(out_w):
            vert_start = i * sh
            vert_end = vert_start + kh
            horiz_start = j * sw
            horiz_end = horiz_start + kw
            patch = padded_images[:, vert_start:vert_end, horiz_start:horiz_end]
            output[:, i, j] = np.sum(patch * kernel, axis=(1, 2))

    return output