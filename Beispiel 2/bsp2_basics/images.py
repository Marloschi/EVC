import numpy as np
import scipy.ndimage
from PIL import Image

import utils


def read_img(inp: str) -> Image.Image:
    """
        Returns a PIL Image given by its input path.
    """
    img = Image.open(inp)
    return img



def convert(img: Image.Image) -> np.ndarray:
    """
        Converts a PIL image [0,255] to a numpy array [0,1].
    """
    ### STUDENT CODE
    # TODO: Implement this function.
    out = np.array(img) / 255.0

    # NOTE: The following lines can be removed. They prevent the framework
    #       from crashing.

    ### END STUDENT CODE
    return out


def switch_channels(img: np.ndarray) -> np.ndarray:
    """
        Swaps the red and green channel of a RGB image given by a numpy array.
    """
    ### STUDENT CODE
    # TODO: Implement this function.
    out = img.copy()
    out[:,:,[0,1]] = out[:,:,[1,0]]

    # NOTE: The following lines can be removed. They prevent the framework
    #       from crashing.

    ### END STUDENT CODE

    return out


def image_mark_green(img: np.ndarray) -> np.ndarray:
    """
        returns a numpy-array (HxW) with 1 where the green channel of the input image is greater or equal than 0.7, otherwise zero.
    """
    ### STUDENT CODE
    # TODO: Implement this function.
    green_channel = img[:,:,1]
    mask = (green_channel >= 0.7)

    # NOTE: The following lines can be removed. They prevent the framework
    #       from crashing.

    ### END STUDENT CODE

    return mask


def image_masked(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
        sets the pixels of the input image to zero where the mask is 1.
    """
    ### STUDENT CODE
    # TODO: Implement this function.
    out = img.copy()
    out[mask == 1] = 0
    # NOTE: The following lines can be removed. They prevent the framework
    #       from crashing.

    ### END STUDENT CODE

    return out


def grayscale(img: np.ndarray) -> np.ndarray:
    """
        Returns a grayscale image of the input. Use utils.rgb2gray().
    """
    ### STUDENT CODE
    # TODO: Implement this function.
    out = img.copy()
    out = utils.rgb2gray(out)

    # NOTE: The following lines can be removed. They prevent the framework
    #       from crashing.

    ### END STUDENT CODE

    return out


def cut_and_reshape(img_gray: np.ndarray) -> np.ndarray:
    """
        Cuts the image in half (x-dim) and stacks it together in y-dim.
    """
    ### STUDENT CODE
    # TODO: Implement this function.
    height,width = img_gray.shape

    left = img_gray[:, :height//2]
    right = img_gray[:, width//2:]
    out = np.vstack((right, left))

    # NOTE: The following lines can be removed. They prevent the framework
    #       from crashing.

    ### END STUDENT CODE

    return out


def filter_image(img: np.ndarray) -> np.ndarray:
    """
        filters the image with the gaussian kernel given below. 
    """
    gaussian = utils.gauss_filter(5, 2)

    ### STUDENT CODE
    # TODO: Implement this function.
    height, width, colors = img.shape
    out = np.zeros(img.shape)
    padded = np.pad(img, ((2, 2), (2, 2), (0, 0)), mode='constant')

    for y in range(height):
        for x in range(width):
            for c in range(colors):
                region = padded[y:y + len(gaussian), x:x + len(gaussian), c]
                out[y, x, c] = np.sum(region * gaussian)
    # NOTE: The following lines can be removed. They prevent the framework
    #       from crashing.


    ### END STUDENT CODE

    return out


def horizontal_edges(img: np.ndarray) -> np.ndarray:
    """
        Defines a sobel kernel to extract horizontal edges and convolves the image with it.
    """
    ### STUDENT CODE
    # TODO: Implement this function.
    sobel = np.array([[1, 2, 1],
                             [0, 0, 0],
                             [-1, -2, -1]])

    # Apply the filter using convolution
    out = scipy.ndimage.correlate(img, sobel, mode='constant')
    # NOTE: The following lines can be removed. They prevent the framework
    #       from crashing.



    ### END STUDENT CODE

    return out
