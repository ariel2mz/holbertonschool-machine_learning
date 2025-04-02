def matrix_addition(mat1, mat2):
    """
    Adds two matrices element-wise.

    Args:
        mat1 (list): The first matrix.
        mat2 (list): The second matrix.

    Returns:
        list: A new matrix that is the result of adding mat1 and mat2 element-wise.
              Returns None if the matrices do not have the same dimensions.
    """
    rows = len(mat1)
    cols = len(mat1[0])
    if rows != len(mat2) or cols != len(mat2[0]):
        return None 

    result = [[0] * cols for _ in range(rows)]

    for i in range(rows):
        for j in range(cols):
            result[i][j] = mat1[i][j] + mat2[i][j]

    return result