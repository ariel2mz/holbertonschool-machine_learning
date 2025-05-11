#!/usr/bin/env python3
"""
no se amigo recien empiezo el codigo
"""
import numpy as np


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """
    sadasdsadsa
    sadsadsad
    """
    decayed_alpha = alpha / (1 + decay_rate * (global_step // decay_step))
    return decayed_alpha
