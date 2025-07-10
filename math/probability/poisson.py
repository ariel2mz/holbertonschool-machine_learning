#!/usr/bin/env python3
"""
sadsadsadsa
dsadsadsadsa
"""
import math


class Poisson:
    """Class that represents a Poisson distribution"""

    def __init__(self, data=None, lambtha=1.):
        """
        Initializes a Poisson distribution.

        Parameters:
        - data (list): sadadasdsa.
        - lambtha (float): assasa.
        """
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = float(sum(data) / len(data))

    def pmf(self, k):
        """
        Parameters:
        - k (int): Number of "successes"

        Returns:
        - PMF value for k (float)
        """
        if not isinstance(k, int):
            k = int(k)
        if k < 0:
            return 0

        e = math.exp(-self.lambtha)
        fac = math.factorial(k)
        lambthak = self.lambtha ** k
        return (lambthak * e) / fac