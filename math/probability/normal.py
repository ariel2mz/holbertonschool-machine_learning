#!/usr/bin/env python3
"""
sadsadsadsa
dsadsadsadsa
"""


class Normal:
    """sadsadsadsads"""

    def __init__(self, data=None, mean=0., stddev=1.):
        """
        asdsadsadsadsa
        """
        if data is None:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            self.mean = float(mean)
            self.stddev = float(stddev)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            n = len(data)
            self.mean = sum(data) / n

            var = 0
            for x in data:
                var += (x - self.mean) ** 2
            var /= n

            self.stddev = var ** 0.5

    def z_score(self, x):
        """
        sadsadsadsadsa
        """
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """
        asdsadasdsa
        """
        return self.mean + z * self.stddev
