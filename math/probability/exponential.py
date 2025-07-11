#!/usr/bin/env python3
"""
sadsadsadsa
dsadsadsadsa
"""


class Exponential:
    """asdsadsadas"""

    def __init__(self, data=None, lambtha=1.):
        """
        asdasdasdasd
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
            mean = sum(data) / len(data)
            if mean <= 0:
                raise ValueError("Mean of data must be positive")
            self.lambtha = 1 / mean
