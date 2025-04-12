#!/usr/bin/env python3
"""
sadsadsa
"""


def poly_integral(poly, C=0):
    """
    Esta descripcion es placeholder no significa nada.

    Args:
        recibe algun argumento seguramente.

    Returns:
        algo retorna, y si no retorna algo retorna none
    """
    if not isinstance(poly, list):
        return None
    if not isinstance(C, int):
        return None

    largo = len(poly)

    if largo == 0:
        return None
    if poly == [0]:
        nuevo = [C]
        return nuevo

    nuevo = [0] * (largo + 1)
    i = 0
    nuevo[0] = C
    for i in range(largo):
        if i == 0:
            nuevo[1] = poly[i]
        else:
            nuevo[i + 1] = poly[i] / (i + 1)
    i = 0
    for i in range(len(nuevo)):
        if nuevo[i] % 1 == 0:
            nuevo[i] = int(nuevo[i])
        if nuevo[i] == 0.0:
            nuevo[i] = 0

    return nuevo
