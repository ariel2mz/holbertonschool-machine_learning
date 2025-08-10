#!/usr/bin/env python3
"""sadasdsadsa"""
import numpy as np


def viterbi(Observation, Emission, Transition, Initial):
    """
    asdsadsadas
    """
    if (not isinstance(Observation, np.ndarray) or Observation.ndim != 1 or
            not isinstance(Emission, np.ndarray) or Emission.ndim != 2 or
            not isinstance(Transition, np.ndarray) or Transition.ndim != 2 or
            not isinstance(Initial, np.ndarray) or Initial.ndim != 2):
        return None, None

    Om = Observation
    Em = Emission
    Tm = Transition
    In = Initial
    T = Om.shape[0]
    N, M = Em.shape

    if Tm.shape != (N, N) or In.shape != (N, 1):
        return None, None

    if not np.allclose(Tm.sum(axis=1), 1):
        return None, None
    if not np.allclose(Em.sum(axis=1), 1):
        return None, None
    if not np.isclose(In.sum(), 1):
        return None, None

    delta = np.zeros((N, T))
    psi = np.zeros((N, T), dtype=int)

    # internet me dio esta funcion si esta mal es culpa
    # de internet
    for i in range(N):
        delta[i, 0] = In[i, 0] * Em[i, Om[0]]
        psi[i, 0] = 0
    for t in range(1, T):
        for j in range(N):
            maxprob = -1
            maxstate = 0
            for i in range(N):
                prob = delta[i, t-1] * Tm[i, j]
                if prob > maxprob:
                    maxprob = prob
                    maxstate = i
            delta[j, t] = maxprob * Em[j, Om[t]]
            psi[j, t] = maxstate
    path = [0] * T
    P = np.max(delta[:, T-1])
    last_state = np.argmax(delta[:, T-1])
    path[T-1] = last_state

    # Inicia la prob más alta de empezar en cada estado (delta)
    # usando la prob inicial y la de emitir la primera observación.
    # luego, para cada tiempo t, calcula para cada
    # estado j cuál es el estado anterior i
    # que maximiza la prob de llegar hasta ahí (maxprob) usando delta y la Tm.
    # guarda el "camino" más prob en psi para poder reconstruirlo después.
    # busca el estado final con mayor prob y retrocede usando psi
    # para obtener la secuencia completa de estados más prob (path).

    for t in range(T-2, -1, -1):
        path[t] = psi[path[t+1], t+1]

    return path, P
