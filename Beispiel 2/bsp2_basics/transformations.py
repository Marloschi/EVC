from typing import List

import numpy as np
import matplotlib.pyplot as plt


def define_transformations() -> List[np.ndarray]:
    """
        Returns the four transformations t1, .., t4 to transform the quadrat. 
        The transformations are determined by using mscale, mrotate and mtranslate.
    """
    ### STUDENT CODE
    # TODO: Implement this function.
    t1 = mtranslate(-3, 0) @ mrotate(55)
    t2 = mtranslate(-3 * np.cos(np.deg2rad(55)), -3 * np.sin(np.deg2rad(55))) @ mrotate(55)
    t3 = mtranslate(3, 1) @ mrotate(70) @ mscale(3, 2)
    t4 = mscale(1, 3) @ mrotate(45)
    # NOTE: The following lines can be removed. They prevent the framework
    #       from crashing.



    ### END STUDENT CODE

    return [t1, t2, t3, t4]


def mscale(sx: float, sy: float) -> np.ndarray:
    """
        Defines a scale matrix. The scales are determined by sx in x and sy in y dimension.
    """
    ### STUDENT CODE
    # TODO: Implement this function.
    m = np.array([
        [sx, 0, 0],
        [0, sy, 0],
        [0, 0, 1]
    ])
    # NOTE: The following lines can be removed. They prevent the framework
    #       from crashing.

    ### END STUDENT CODE

    return m


def mrotate(angle: float) -> np.ndarray:
    """
        Defines a rotation matrix (z-axis) determined by the angle in degree (!).
    """
    ### STUDENT CODE
    # TODO: Implement this function.
    theta = np.radians(angle)
    cos = np.cos(theta)
    sin = np.sin(theta)
    m = np.array([
        [cos, -sin, 0],
        [sin, cos, 0],
        [0, 0, 1]
    ])
    # NOTE: The following lines can be removed. They prevent the framework
    #       from crashing.



    ### END STUDENT CODE

    return m


def mtranslate(tx: float, ty: float) -> np.ndarray:
    """
        Defines a translation matrix. tx in x, ty in y direction.
    """
    ### STUDENT CODE
    # TODO: Implement this function.
    m = np.array([
        [1, 0, tx],
        [0, 1, ty],
        [0, 0, 1]
    ])
    # NOTE: The following lines can be removed. They prevent the framework
    #       from crashing.



    ### END STUDENT CODE

    return m


def transform_vertices(v: np.ndarray, m: np.ndarray) -> np.ndarray:
    """
        transform the (3xN) vertices given by v with the (3x3) transformation matrix determined by m.
    """
    ### STUDENT CODE
    # TODO: Implement this function.
    out = m @ v
    # NOTE: The following lines can be removed. They prevent the framework
    #       from crashing.



    ### END STUDENT CODE

    return out


def display_vertices(v: np.ndarray, title: str) -> None:
    """
        Plot the vertices in a matplotlib figure.
    """
    # create the figure and set the title
    plt.figure()
    plt.axis('square')

    plt.title(title)

    # x and y limits
    plt.xlim((-6, 6))
    plt.ylim((-6, 6))
    plt.xticks(range(-6, 6))
    plt.yticks(range(-6, 6))

    # plot coordinate axis
    plt.axvline(color='black')
    plt.axhline(color='black')
    plt.grid()

    # we just add the last element, so plot can do our job :)
    v_ = np.concatenate((v, v[:, 0].reshape(3, -1)), axis=1)

    plt.plot(v_[0, :], v_[1, :], linewidth=3)
    plt.show()
