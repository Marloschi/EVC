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
    height, width, channels = img.shape
    out = np.zeros_like(img)

    # Kernel size and radius
    k_size = 5
    k_radius = k_size // 2

    # Iterate over each pixel in the image
    for y in range(height):
        for x in range(width):
            # Initialize sum for each channel
            pixel_sum = [0.0, 0.0, 0.0]

            # Iterate over kernel
            for ky in range(-k_radius, k_radius + 1):
                for kx in range(-k_radius, k_radius + 1):
                    # Calculate image coordinates
                    img_y = y + ky
                    img_x = x + kx

                    # Check if within image bounds
                    if (0 <= img_y < height) and (0 <= img_x < width):
                        # Multiply and accumulate for each channel
                        for c in range(channels):
                            pixel_sum[c] += img[img_y, img_x, c] * gaussian[ky + k_radius, kx + k_radius]
                    # Else: treat as [0,0,0] (no addition needed)

            # Assign the filtered value to output
            for c in range(channels):
                out[y, x, c] = pixel_sum[c]
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
    G_horizontal = np.array([[1, 2, 1],
                             [0, 0, 0],
                             [-1, -2, -1]])

    # Apply the filter using convolution
    out = scipy.ndimage.correlate(img, G_horizontal, mode='constant', cval=0)
    # NOTE: The following lines can be removed. They prevent the framework
    #       from crashing.



    ### END STUDENT CODE

    return out
