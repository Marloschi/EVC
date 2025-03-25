from typing import List, Tuple
import numpy as np


def define_triangle() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    ### STUDENT CODE
    # TODO: Implement this function.

    A, B, C, D, E, F, G, H = 1, 2, 4, 2, 6, 8, 5, 5

    P1 = np.array([(1 + C), -(1 + A), -(1 + E)])
    P2 = np.array([-(1 + G), -(1 + B), (1 + H)])
    P3 = np.array([-(1 + D), (1 + F), -(1 + B)])

    # NOTE: The following lines can be removed. They prevent the framework
    #       from crashing.


    ### END STUDENT CODE

    return P1, P2, P3


def define_triangle_vertices(P1: np.ndarray, P2: np.ndarray, P3: np.ndarray) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray]:
    ### STUDENT CODE
    # TODO: Implement this function.
    P1P2 = P2 - P1
    P2P3 = P3 - P2
    P3P1 = P1 - P3
    # NOTE: The following lines can be removed. They prevent the framework
    #       from crashing.

    ### END STUDENT CODE

    return P1P2, P2P3, P3P1


def compute_lengths(P1P2: np.ndarray, P2P3: np.ndarray, P3P1: np.ndarray) -> List[float]:
    ### STUDENT CODE
    # TODO: Implement this function.

    P1P2Length = np.sqrt(P1P2**2)
    P2P3Length = np.sqrt(P2P3**2)
    P3P1Length = np.sqrt(P3P1**2)
    norms = [P1P2Length, P2P3Length, P3P1Length]
    # NOTE: The following lines can be removed. They prevent the framework
    #       from crashing.

    ### END STUDENT CODE

    return norms


def compute_normal_vector(P1P2: np.ndarray, P2P3: np.ndarray, P3P1: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    ### STUDENT CODE
    # TODO: Implement this function.
    n = np.cross(P1P2, P2P3)
    norm = np.linalg.norm(n)
    n_normalized = n/norm if norm != 0 else n
    # NOTE: The following lines can be removed. They prevent the framework
    #       from crashing.


    ### END STUDENT CODE

    return n, n_normalized


def compute_triangle_area(n: np.ndarray) -> float:
    ### STUDENT CODE
    # TODO: Implement this function.
    norm = np.linalg.norm(n)

    area = 0.5*norm if norm >= 0 else 0.5*(norm*(-1))
    # NOTE: The following lines can be removed. They prevent the framework
    #       from crashing.


    ### END STUDENT CODE

    return area


def compute_angles(P1P2: np.ndarray, P2P3: np.ndarray, P3P1: np.ndarray) -> Tuple[float, float, float]:
    ### STUDENT CODE
    # TODO: Implement this function.
    cos_alpha = np.dot(-P3P1, P1P2) / (np.linalg.norm(P3P1) * np.linalg.norm(P1P2))
    alpha = np.degrees(np.arccos(np.clip(cos_alpha, -1.0, 1.0)))

    cos_beta = np.dot(-P1P2, P2P3) / (np.linalg.norm(P1P2) * np.linalg.norm(P2P3))
    beta = np.degrees(np.arccos(np.clip(cos_beta, -1.0, 1.0)))

    cos_gamma = np.dot(P2P3, -P3P1) / (np.linalg.norm(P2P3) * np.linalg.norm(P3P1))
    gamma = np.degrees(np.arccos(np.clip(cos_gamma, -1.0, 1.0)))
    # NOTE: The following lines can be removed. They prevent the framework
    #       from crashing.

    ### END STUDENT CODE

    return alpha, beta, gamma
