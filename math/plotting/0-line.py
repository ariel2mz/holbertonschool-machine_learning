#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
"""
sadsadsadsa
"""


def line():
    """
    sadsadsadsa
    """
    y = np.arange(0, 11) ** 3
    plt.figure(figsize=(6.4, 4.8))

    x = np.arange(0, 11)  # Eje x desde 0 hasta 10
    plt.plot(x, y, 'r-')  # 'r-' = línea roja sólida
    plt.xlim(0, 10)       # Limitar el eje x de 0 a 10
    plt.show()
