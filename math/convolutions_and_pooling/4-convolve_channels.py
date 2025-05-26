"""
sadsdadsa
sadsaddsa
"""
import numpy as np


def convolve_channels(images, kernel, padding='same', stride=(1, 1)):
    """
    Performs a convolution on images with multiple channels

    Args:
        images: np.ndarray of shape (m, h, w, c)
        kernel: np.ndarray of shape (kh, kw, c)
        padding: 'same', 'valid', or (ph, pw)
        stride: tuple (sh, sw)

    Returns:
        np.ndarray containing the convolved images
    """
    m, h, w, c = images.shape
    kh, kw, kc = kernel.shape
    sh, sw = stride

    if padding == 'same':
        ph = ((h - 1) * sh + kh - h + 1) // 2
        pw = ((w - 1) * sw + kw - w + 1) // 2
    elif padding == 'valid':
        ph = pw = 0
    else:
        ph, pw = padding

    padd = np.pad(
        images,
        pad_width=((0, 0), (ph, ph), (pw, pw), (0, 0)),
        mode='constant',
        constant_values=0
    )

    nuevoh = (h + 2 * ph - kh) // sh + 1
    nuevow = (w + 2 * pw - kw) // sw + 1

    nuevo = np.zeros((m, nuevoh, nuevow))

    for i in range(nuevoh):
        hstart = i * sh
        for j in range(nuevow):
            wstart = j * sw
            region = padd[:, hstart:hstart+kh, wstart:wstart+kw, :]
            nuevo[:, i, j] = np.sum(region * kernel, axis=(1, 2, 3))

    return nuevo
