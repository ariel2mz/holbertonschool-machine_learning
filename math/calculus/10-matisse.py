#!/usr/bin/env python3
"""
sadsadsa
"""


def poly_derivative(poly):
    """
    Esta descripcion es placeholder no significa nada.

    Args:
        recibe algun argumento seguramente.

    Returns:
        algo retorna, y si no retorna algo retorna none
    """
    if not isinstance(poly, list):
        return None

    largo = len(poly)

    if largo == 0:
        return None
    if largo == 1:
        return [0]

    nuevo = [0] * (largo - 1)
    i = 1
    for i in range(0, largo):
        nuevo[i - 1] = poly[i] * i
    return nuevo
