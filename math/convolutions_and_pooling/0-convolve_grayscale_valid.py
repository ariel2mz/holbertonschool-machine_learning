#!/usr/bin/env python3
"""
https://www.youtube.com/watch?v=a6axZGuAs3o&ab_channel=ShittyRoboticsTeacher
este viideo lo explica re  bien tio
"""
import numpy as np


def convolve_grayscale_valid(images, kernel):
    """
    Performs a valid convolution on grayscale images.

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

    nuevoh = h - kh + 1
    nuevow = w - kw + 1
    nuevo = np.zeros((m, nuevoh, nuevow))

    for i in range(nuevoh):
        for j in range(nuevow):
            porcion = images[:, i:i+kh, j:j+kw]
            pmulti = porcion * kernel
            psumad = np.sum(pmulti, axis=(1, 2))
            nuevo[:, i, j] = psumad

    return nuevo
