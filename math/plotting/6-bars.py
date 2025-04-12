#!/usr/bin/env python3
"""
sadsadsadsa
dsadsadsadsa
"""
import numpy as np
import matplotlib.pyplot as plt


def bars():
    """
    Esta descripcion es placeholder no significa nada.

    Args:
        recibe algun argumento seguramente.

    Returns:
        algo retorna, y si no retorna algo retorna none
    """

    np.random.seed(5)
    fruit = np.random.randint(0, 20, (4, 3))
    plt.figure(figsize=(6.4, 4.8))

    labels = ['apples', 'bananas', 'oranges', 'peaches']
    colors = ['red', 'yellow', '#ff8000', '#ffe5b4']
    persons = ['Farrah', 'Fred', 'Felicia']
    width = 0.5

    bottom = np.zeros(len(persons))

    for i in range(fruit.shape[0]):
        plt.bar(persons, fruit[i],
                width, bottom=bottom,
                color=colors[i], label=labels[i])
        bottom += fruit[i]

    plt.ylabel("Quantity of Fruit")
    plt.ylim(0, 80)
    plt.yticks(np.arange(0, 81, 10))
    plt.title("Number of Fruit per Person")
    plt.legend()

    plt.show()
