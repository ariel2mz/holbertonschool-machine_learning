#!/usr/bin/env python3
"""fghkl"""
import numpy as np
GP = __import__('2-gp').GaussianProcess

class BayesianOptimization:
    """
    dfghjkl
    """
    def __init__(self, f, X_init, Y_init, bounds, ac_samples,
                 l=1, sigma_f=1, xsi=0.01, minimize=True):
        """
        dfghljfdfjk
        """
        self.f = f
        self.gp = GP(X_init, Y_init, l=l, sigma_f=sigma_f)
        self.X_s = np.linspace(bounds[0],
                               bounds[1], ac_samples).reshape(-1, 1)
        self.xsi = xsi
        self.minimize = minimize

    def _norm_pdf(self, x):
        """Standard normal PDF"""
        return (1.0 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * x**2)

    def _norm_cdf(self, x):
        """Standard normal CDF using erf"""
        return 0.5 * (1 + np.erf(x / np.sqrt(2)))

    def acquisition(self):
        """
        Calculates the next best sample location using the Expected Improvement (EI) method.
        Returns:
            X_next: numpy.ndarray of shape (1,) representing the next best sample point
            EI: numpy.ndarray of shape (ac_samples,) containing the expected improvement
        """
        mu, sigma = self.gp.predict(self.X_s)
        sigma = sigma.reshape(-1, 1)

        if self.minimize:
            Y_best = np.min(self.gp.Y)
            improvement = Y_best - mu - self.xsi
        else:
            Y_best = np.max(self.gp.Y)
            improvement = mu - Y_best - self.xsi

        with np.errstate(divide='warn'):
            Z = np.zeros_like(improvement)
            mask = sigma != 0
            Z[mask] = improvement[mask] / sigma[mask]

            EI = np.zeros_like(improvement)
            EI[mask] = improvement[mask] * self._norm_cdf(Z[mask]) + sigma[mask] * self._norm_pdf(Z[mask])

        X_next = self.X_s[np.argmax(EI)]
        return X_next, EI.ravel()

    def optimize(self, iterations=100):
        """
        Performs Bayesian optimization to find the optimal point of the black-box function.
        
        Args:
            iterations (int): Maximum number of iterations to perform.
        
        Returns:
            X_opt (numpy.ndarray): Shape (1,) optimal point found.
            Y_opt (numpy.ndarray): Shape (1,) function value at X_opt.
        """
        for _ in range(iterations):
            X_next, _ = self.acquisition()
            if np.any(np.isclose(self.gp.X, X_next)):
                break
            Y_next = self.f(X_next)
            self.gp.update(X_next, Y_next)

        if self.minimize:
            idx_opt = np.argmin(self.gp.Y)
        else:
            idx_opt = np.argmax(self.gp.Y)

        X_opt = self.gp.X[idx_opt]
        Y_opt = self.gp.Y[idx_opt]

        return X_opt.reshape(1,), Y_opt.reshape(1,)
