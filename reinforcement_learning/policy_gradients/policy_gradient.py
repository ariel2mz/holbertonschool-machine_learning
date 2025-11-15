#!/usr/bin/env python3
"""
asgsagsagsaa
"""
import numpy as np


def policy(matrix, weight):
    """
    askaslkfasklfsa
    """
    z = matrix @ weight

    exp = np.exp(z - np.max(z))

    prob = exp / np.sum(exp)

    return prob


def policy_gradient(state, weight):
    """
    fsafsafsafsa
    """

    state = state.reshape(1, -1)

    prob = policy(state, weight)

    act = np.random.choice(prob.shape[1], p=prob[0])

    onehot = np.zeros(prob.shape[1])
    onehot[act] = 1

    grad = state.T @ (onehot - prob)

    return int(act), grad

    """
    Esta función ayuda a un agente de inteligencia
    artificial a aprender por prueba y error.
    Imagina que el agente está en un videojuego y necesita decidir qué
    movimiento hacer.
    Primero, la función mira el estado actual del juego
    (la pantalla, la posición, etc.)
    y calcula qué tan probable es que cada movimiento sea bueno usando
    los pesos actuales de la red neuronal.
    Es como decir: "basado en lo que he aprendido hasta ahora,
    tengo 60% de probabilidades de moverme a la izquierda y 40%
    a la derecha".
    Luego, no elige siempre el movimiento más probable,
    sino que tira un dado virtual
    para elegir uno al azar,
    pero dando más peso a los movimientos que parecen mejores.
    Esto le permite explorar
    y descubrir nuevas estrategias.
    Después viene la parte del aprendizaje:
    la función crea una matriz de ajustes
    y marca con un check la acción que eligió, como diciendo
    "esta acción fue la que tomé".
    Finalmente, para cada movimiento posible,
    le resta un pequeño castigo proporcional a
    qué tan probable era ese movimiento.
    Si un movimiento era muy tentador pero no lo elegí,
    lo castigo más. Si era poco
    probable,
    casi no lo castigo.
    El resultado es una guía que le dice a la red neuronal:
    "ajusta tus conexiones de esta manera para que la próxima
    vez tengas más probabilidades de elegir buenos movimientos y menos
    de elegir malos".
    """
