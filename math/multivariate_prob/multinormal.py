#!/usr/bin/env python3
"""
asdsaddas
"""
import numpy as np


class MultiNormal:
    """
    Represents a Multivariate Normal distribution.
    """

    def __init__(self, data):
        """
        Initialize the MultiNormal class.

        Parameters:
        data (numpy.ndarray): Data of shape (d, n) where:
            - n is the number of data points
            - d is the number of dimensions

        Raises:
        TypeError: If data is not a 2D numpy.ndarray
        ValueError: If n is less than 2
        """
        if not isinstance(data, np.ndarray):
            raise TypeError("data must be a 2D numpy.ndarray")
        if data.ndim != 2:
            raise TypeError("data must be a 2D numpy.ndarray")
        d, n = data.shape
        if n < 2:
            raise ValueError("data must contain multiple data points")
        self.mean = np.mean(data, axis=1, keepdims=True)
        centered_data = data - self.mean
        self.cov = (1 / (n - 1)) * np.dot(centered_data, centered_data.T)

    def pdf(self, x):
        """
        Calculate the Probability Density Function at a data point x.

        Parameters:
        x (numpy.ndarray): Data point of shape (d, 1)

        Returns:
        float: Value of the PDF at x

        Raises:
        TypeError: If x is not a numpy.ndarray
        ValueError: If x does not have shape (d, 1)
        """
        if not isinstance(x, np.ndarray):
            raise TypeError("x must be a numpy.ndarray")

        if x.shape != (self.d, 1):
            raise ValueError(f"x must have the shape ({self.d}, 1)")

        d = self.d
        pi = np.pi

        det_cov = np.linalg.det(self.cov)

        two_pi_pow = (2 * pi) ** (d / 2)
        normalization = 1 / (two_pi_pow * np.sqrt(det_cov))

        diff = x - self.mean

        inv_cov = np.linalg.inv(self.cov)

        quadratic = np.dot(np.dot(diff.T, inv_cov), diff)

        quadratic_scalar = quadratic[0, 0]

        exponent = np.exp(-0.5 * quadratic_scalar)

        pdf_value = normalization * exponent

        return pdf_value
