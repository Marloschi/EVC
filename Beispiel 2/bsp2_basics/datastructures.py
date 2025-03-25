from typing import Tuple
import numpy as np



def define_structures() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
        Defines the two vectors v1 and v2 as well as the matrix M determined by your matriculation number.
    """
    ### STUDENT CODE
    # TODO: Implement this function.
    # 12426855
    # ABCDEFGH
    M = np.array([[2, 2, 4], [2, 5, 1], [6, 5, 8]])
    v1 = np.array([2, 1, 4])
    v2 = np.array([8, 2, 6])

    # NOTE: The following lines can be removed. They prevent the framework
    #       from crashing.

    # v1 = np.zeros(3)
    # v2 = v1.copy()
    # M = np.zeros((3,3))

    ### END STUDENT CODE

    return v1, v2, M


def sequence(M: np.ndarray) -> np.ndarray:
    """
        Defines a vector given by the minimum and maximum digit of your matriculation number. Step size = 0.25.
    """
    ### STUDENT CODE
    # TODO: Implement this function.

    min = np.min(M)
    max = np.max(M)

    result = np.arange(min, max + 0.25, 0.25)

    # NOTE: The following lines can be removed. They prevent the framework
    #       from crashing.

    # result = np.zeros(10)

    ### END STUDENT CODE

    return result


def matrix(M: np.ndarray) -> np.ndarray:
    """
        Defines the 15x9 block matrix as described in the task description.
    """
    ### STUDENT CODE
    # TODO: Implement this function.

    whiteField = np.zeros((3, 3))
    blackField = M

    firstAndLastRow = np.vstack([np.vstack([np.vstack([blackField, whiteField]), blackField]), whiteField])
    secondRow = np.vstack([np.vstack([np.vstack([whiteField, blackField]), whiteField]), blackField])

    r = np.hstack([np.hstack([firstAndLastRow, secondRow]), firstAndLastRow])

    # NOTE: The following lines can be removed. They prevent the framework
    #       from crashing.

    ### END STUDENT CODE

    return r


def dot_product(v1: np.ndarray, v2: np.ndarray) -> float:
    """
        Dot product of v1 and v2.
    """
    ### STUDENT CODE
    # TODO: Implement this function.
    #r = np.dot(v1, v2)
    r = sum(v1[i]*v2[i] for i in range(len(v1)))



    # NOTE: The following lines can be removed. They prevent the framework
    #       from crashing.



    ### END STUDENT CODE

    return r


def cross_product(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """
        Cross product of v1 and v2.
    """
    ### STUDENT CODE
    # TODO: Implement this function.
    #r = np.cross(v1, v2)
    r = np.array([
        v1[1] * v2[2] - v1[2] * v2[1],
        v1[2] * v2[0] - v1[0] * v2[2],
        v1[0] * v2[1] - v1[1] * v2[0]
    ])
    # NOTE: The following lines can be removed. They prevent the framework
    #       from crashing.


    ### END STUDENT CODE

    return r


def vector_X_matrix(v: np.ndarray, M: np.ndarray) -> np.ndarray:
    """
        Defines the vector-matrix multiplication v*M.
    """
    ### STUDENT CODE
    # TODO: Implement this function.

    r = np.zeros(3)
    for i in range(len(v)):
        for j in range(len(M)):
            r[i] += v[j]*M[j][i]
    # NOTE: The following lines can be removed. They prevent the framework
    #       from crashing.

    ### END STUDENT CODE

    return r


def matrix_X_vector(M: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
        Defines the matrix-vector multiplication M*v.
    """
    ### STUDENT CODE
    # TODO: Implement this function.
    r = np.zeros(3)
    for i in range(len(v)):
        for j in range(len(M)):
            r[i] += M[j][i]*v[j]
    # NOTE: The following lines can be removed. They prevent the framework
    #       from crashing.

    ### END STUDENT CODE

    return r


def matrix_X_matrix(M1: np.ndarray, M2: np.ndarray) -> np.ndarray:
    """
        Defines the matrix multiplication M1*M2.
    """
    ### STUDENT CODE
    # TODO: Implement this function.
    r = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                r[i][j] += M1[i][k] * M2[k][j]
    # NOTE: The following lines can be removed. They prevent the framework
    #       from crashing.

    ### END STUDENT CODE

    return r


def matrix_Xc_matrix(M1: np.ndarray, M2: np.ndarray) -> np.ndarray:
    """
        Defines the element-wise matrix multiplication M1*M2 (Hadamard Product).
    """
    ### STUDENT CODE
    # TODO: Implement this function.
    r = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            r[i][j] = M1[i][j] * M2[i][j]

    # NOTE: The following lines can be removed. They prevent the framework
    #       from crashing.
    r = np.zeros(M1.shape)
    ### END STUDENT CODE

    return r
