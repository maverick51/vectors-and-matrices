import math


class ShapeException(Exception):
    """Handles exception for shapes of vectors and matrices"""
    pass


def vector_add(vector1, vector2):
    """Adds two vectors together and returns the result"""
    shape_vector1 = shape(vector1)
    shape_vector2 = shape(vector2)
    if shape_vector1 != shape_vector2:
        raise ShapeException("Error, vector shapes are not equal so cannot add together.")
    return [vector1[x] + vector2[x] for x in range(len(vector2))]


def vector_sub(vector1, vector2):
    """Subtracts two vectors and returns the result"""
    shape_vector1 = shape(vector1)
    shape_vector2 = shape(vector2)
    if shape_vector1 != shape_vector2:
        raise ShapeException("Error, vector shapes are not equal so cannot subtract together.")
    return [vector1[x] - vector2[x] for x in range(len(vector2))]


def vector_sum(*args):
    """Sums up a varying number of vectors"""
    count = 0
    list_to_sum = []
    the_same = False
    while count <= len(args) - 1:
        list_to_sum[count] = shape(args[count])
        count += 1
    while count <= len(list_to_sum) - 1:
        if list_to_sum[count] == list_to_sum[count+1]:
            the_same = True
    if not the_same:
        raise ShapeException("Error, vector shapes are not equal so cannot sum together.")
    # TO DO - find the sum of the vectors


def vector_multiply(vector1, number):
    """Multiply a vector by a scalar and return the result"""
    return [number * vector1[x] for x in range(len(vector1))]


def vector_mean():
    pass
    # TO DO


def dot(vector1, vector2):
    """Multiplies two vectors together and returns a scalar"""
    shape_vector1 = shape(vector1)
    shape_vector2 = shape(vector2)
    if shape_vector1 != shape_vector2:
        raise ShapeException("Error, vector shapes are not equal so cannot multiply together.")
    return [vector1[x] * vector2[x] for x in range(len(vector1))]


def magnitude(vector):
    """Finds the magnitude of the vector and returns the result"""
    return math.sqrt(vector[0] * vector[0] + vector[1] * vector[1])


def matrix_addition(matrix1, matrix2):
    """Adds two matrices together and returns the result"""
    return [matrix1[x] + matrix2[x] for x in range(len(matrix1))]


def matrix_subtraction(matrix1, matrix2):
    """Subtracts two matrices together and returns the result"""
    return [matrix1[x] - matrix2[x] for x in range(len(matrix1))]


def matrix_scalar_multiply(matrix, scalar):
    """Multiplies a matrix by a scalar and returns a matrix"""
    return [scalar * matrix[x] for x in range(len(matrix))]


def matrix_vector_multiply(matrix, vector):
    """Multiplies a matrix by a vector and returns the matrix"""
    shape_vector1 = shape(matrix)
    shape_vector2 = shape(vector)
    if shape_vector1 != shape_vector2:
        raise ShapeException("Error, vector/matrix shapes are not equal so cannot multiply together.")
    return [matrix[x] * vector[x] for x in range(len(vector))]


def matrix_matrix_multiply(matrix1, matrix2):
    """Multiplies two matrices together and returns a matrix"""
    matrix1_col = matrix_col(matrix1, shape(matrix1))
    matrix2_row = matrix_row(matrix2, shape(matrix2))
    if matrix1_col != matrix2_row:
        raise ShapeException("Error, matrix col/row shapes are not equal so cannot multiply together.")
    return [matrix1[x] * matrix2[x] for x in range(len(matrix1))]


def shape(vector_or_matrix):
    """Returns the size of the vector or matrix"""
    return len(vector_or_matrix),


def matrix_row(matrix, row):
    """Returns the row from the matrix given"""
    return matrix[row]


def matrix_col(matrix, col):
    """Returns the col from the matrix given"""
    return [matrix[x] for x in range(len(matrix))]


if __name__ == '__main__':
    vector_a = [1, 2]
    vector_b = [3, 4]
    vector_c = [1, 2, 3]
    vector_d = [4, 5, 6]
    matrix_a = [[1, 2],
                [3, 4]]
    matrix_b = [[5, 6],
                [7, 8]]
    matrix_c = [[1, 2, 3],
                [4, 5, 6]]
    matrix_d = [[1, 3, 5],
                [2, 4, 6]]
    matrix_e = [[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]]
    matrix_f = [[9, 8, 7],
                [6, 5, 4],
                [3, 2, 1]]