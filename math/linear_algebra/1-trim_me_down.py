#!/usr/bin/env python3
matrix = [[1, 3, 9, 4, 5, 8], [2, 4, 7, 3, 4, 0], [0, 3, 4, 6, 1, 5]]
the_middle = []
for n in matrix:
    the_middle.append([n[2], n[3]])
print("The middle columns of the matrix are: {}".format(the_middle))
