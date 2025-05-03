#!/usr/bin/env python3

import numpy as np
one_hot = __import__('3-one_hot').one_hot

if __name__ == '__main__':
    labels = [5, 0, 4, 1, 9, 2, 1, 3, 1, 4]
    print(labels)
    print(one_hot(labels))   
