#!/usr/bin/env python3
import numpy as np
import tensorflow as tf


class NST:
    """Performs tasks for neural style transfer"""


    style_layers = [
        'block1_conv1',
        'block2_conv1',
        'block3_conv1',
        'block4_conv1',
        'block5_conv1'
    ]
    content_layer = 'block5_conv2'

    def __init__(self, style_image: np.ndarray, content_image: np.ndarray,
                 alpha: float = 1e4, beta: float = 1):
        """Class constructor.

        Args:
            style_image (np.ndarray): sadsadsadsa
            content_image (np.ndarray): sadsadsa
            alpha (float, optional): sadsadsadsadsa
            beta (float, optional): sadsadsa

        Raises:
            TypeError: sadsadsads
        """

        self._validate_image(style_image, 'style_image')
        self._validate_image(content_image, 'content_image')

        if not isinstance(alpha, (int, float)) or alpha < 0:
            raise TypeError('alpha must be a non-negative number')
        if not isinstance(beta, (int, float)) or beta < 0:
            raise TypeError('beta must be a non-negative number')

        self.style_image = self._preprocess_image(style_image)
        self.content_image = self._preprocess_image(content_image)

        self.alpha = alpha
        self.beta = beta