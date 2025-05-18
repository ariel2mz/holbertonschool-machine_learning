#!/usr/bin/env python3
import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """
    Conduct forward propagation using Dropout.

    Parameters:
    - X: input data (nx, m)
    - weights: dictionary of weights and biases
    - L: number of layers
    - keep_prob: probability of keeping a neuron active

    Returns:
    - cache: dictionary with layer activations and dropout masks
    """
    cache = {}
    cache['A0'] = X

    for i in range(1, L + 1):
        W = weights['W' + str(i)]
        b = weights['b' + str(i)]
        A_prev = cache['A' + str(i - 1)]

        Z = np.matmul(W, A_prev) + b

        if i == L:
            # a la ultima layer se le hace softmax
            """
            La función softmax transforma un vector de valores reales
            (por ejemplo, las salidas de la última capa de una red neuronal)
            en una distribución de probabilidad. Es decir, convierte
            esos valores en números entre 0 y 1 que suman 1 y
            pueden interpretarse como
            probabilidades.
            atentamente: chat gepete
            """
            exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
            A = exp_Z / np.sum(exp_Z, axis=0, keepdims=True)
        else:
            A = np.tanh(Z)
            D = np.random.rand(A.shape[0], A.shape[1]) < keep_prob
            A *= D
            A /= keep_prob
            cache['D' + str(i)] = D

        cache['A' + str(i)] = A

    return cache
