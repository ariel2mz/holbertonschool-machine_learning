#!/usr/bin/env python3
"""
asgsagsagsaa
"""
import pandas as pd

def from_numpy(array):
    """
    sfasafsafsaf
    """
    n_cols = array.shape[1]
    cols = [chr(65 + i) for i in range(n_cols)]
    return pd.DataFrame(array, columns=cols)
