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
    def pdf(self, x):
        """
        ASDFGHJKLKHGF
        """
        pi = 3.1415926536
        e = 2.7182818285
        p1 = 1 / (self.stddev * (2 * pi) ** 0.5)
        exp = -0.5 * ((x - self.mean) / self.stddev) ** 2
        p2 = e ** exp
        return p1 * p2

    def cdf(self, x):
        """Calculates the value of the CDF for a given x-value"""
        z = (x - self.mean) / (self.stddev * (2 ** 0.5))

        # Constants for the approximation
        t = 1 / (1 + 0.3275911 * abs(z))
        a1 = 0.254829592
        a2 = -0.284496736
        a3 = 1.421413741
        a4 = -1.453152027
        a5 = 1.061405429
        e = 2.7182818285

        erf_approx = 1 - (a1 * t + a2 * t**2 + a3 * t**3 + a4 * t**4 + a5 * t**5) * (e ** (-z**2))

        if z >= 0:
            return 0.5 * (1 + erf_approx)
        else:
            return 0.5 * (1 - erf_approx)