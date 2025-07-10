#!/usr/bin/env python3
"""
sadsadsadsa
dsadsadsadsa
"""


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
        - k (int or float): Number of successes

        Returns:
        - PMF value (float)
        """
        if not isinstance(k, int):
            k = int(k)
        if k < 0:
            return 0

        e = 2.7182818285

        fac = 1
        for i in range(1, k + 1):
            fac *= i

        lambthak = self.lambtha ** k
        equis = e ** (-self.lambtha)
        return (lambthak * equis) / fac

    def cdf(self, k):
        """
        Calculates the CDF (Cumulative Distribution Function) for k successes.
        """
        if not isinstance(k, int):
            k = int(k)
        if k < 0:
            return 0

        e = 2.7182818285
        cdfsum = 0

        for i in range(k + 1):
            fac = 1
            for j in range(1, i + 1):
                fac *= j
            cdfsum += (self.lambtha ** i) / fac

        equis = e ** (-self.lambtha)
        return equis * cdfsum
