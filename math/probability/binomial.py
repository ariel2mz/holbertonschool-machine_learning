#!/usr/bin/env python3
"""
sadsadsadsa
dsadsadsadsa
"""

class Binomial:
    """asdsadas"""

    def __init__(self, data=None, n=1, p=0.5):
        """
        asdasdas
        """
        if data is None:
            if n <= 0:
                raise ValueError("n must be a positive value")
            if p <= 0 or p >= 1:
                raise ValueError("p must be greater than 0 and less than 1")
            self.n = int(n)
            self.p = float(p)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            mean = sum(data) / len(data)
            var = sum((x - mean) ** 2 for x in data) / len(data)

            pest = 1 - (var / mean)
            nest = round(mean / pest)
            pest = mean / nest

            self.n = int(nest)
            self.p = float(pest)

    def pmf(self, k):
        """asddasdsadsa"""
        if not isinstance(k, int):
            k = int(k)

        if k < 0 or k > self.n:
            return 0

        def factorial(x):
            result = 1
            for i in range(1, x + 1):
                result *= i
            return result

        nfact = factorial(self.n)
        kfact = factorial(k)
        nkfact = factorial(self.n - k)

        comb = nfact / (kfact * nkfact)
        pt = self.p ** k
        qt = (1 - self.p) ** (self.n - k)

        return comb * pt * qt
    
    def cdf(self, k):
        """asdsadsadsa"""
        if not isinstance(k, int):
            k = int(k)
        if k < 0:
            return 0

        cdfsum = 0
        for i in range(0, min(k + 1, self.n + 1)):
            cdfsum += self.pmf(i)
        return cdfsum