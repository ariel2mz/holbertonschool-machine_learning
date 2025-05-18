#!/usr/bin/env python3
"""
sadasdasdsad
"""


def early_stopping(cost, opt_cost, threshold, patience, count):
    """
    sadsadasda
    sadasdasdsa
    """
    if opt_cost - cost > threshold:
        count = 0
    else:
        count += 1
    stop = count >= patience
    return stop, count
