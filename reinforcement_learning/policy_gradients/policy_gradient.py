#!/usr/bin/env python3
"""
asgsagsagsaa
"""
import numpy as np


def policy(matrix, weight):
    """
    askaslkfasklfsa
    """
    z = matrix.dot(weight)
    expz = np.exp(z - np.max(z, axis=1, keepdims=True))
    prob = expz / np.sum(expz, axis=1, keepdims=True)

    return prob

def policy_gradient(state, weight):
    """
    safsafsafsa
    """

    # ∇logπ(a|s) = φ(s) * (1(a=a') - π(a'|s))
    prob = policy(state, weight)

    act = np.random.choice(len(prob), p=prob)

    grad = np.zeros_like(weight)

    grad[:, act] = state

    for a in range(weight.shape[1]):
        grad[:, a] -= prob[a] * state

    """
    Esta función ayuda a un agente de inteligencia
    artificial a aprender por prueba y error.
    Imagina que el agente está en un videojuego y necesita decidir qué movimiento hacer.
    Primero, la función mira el estado actual del juego (la pantalla, la posición, etc.)
    y calcula qué tan probable es que cada movimiento sea bueno usando
    los pesos actuales de la red neuronal.
    Es como decir: "basado en lo que he aprendido hasta ahora,
    tengo 60% de probabilidades de moverme a la izquierda y 40%
    a la derecha".
    Luego, no elige siempre el movimiento más probable, sino que tira un dado virtual
    para elegir uno al azar, pero dando más peso a los movimientos que parecen mejores.
    Esto le permite explorar
    y descubrir nuevas estrategias.
    Después viene la parte del aprendizaje: la función crea una matriz de ajustes
    y marca con un check la acción que eligió, como diciendo
    "esta acción fue la que tomé".
    Finalmente, para cada movimiento posible,
    le resta un pequeño castigo proporcional a qué tan probable era ese movimiento.
    Si un movimiento era muy tentador pero no lo elegí, lo castigo más. Si era poco
    probable,
    casi no lo castigo.
    El resultado es una guía que le dice a la red neuronal:
    "ajusta tus conexiones de esta manera para que la próxima
    vez tengas más probabilidades de elegir buenos movimientos y menos
    de elegir malos".
    """
    return act, grad
