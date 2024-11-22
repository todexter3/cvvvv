''' 
This file is part of the code for Part 1:
    It contains a function get_filters(), which generates a set of filters in the
format of matrices. (Hint: You add more filters, like the Dirac delta function, whose response is the intensity of the pixel itself.)
'''
import numpy as np
import math


def get_filters():
    """
    Define a set of filters in the form of matrices.

    Returns:
        list: A list of filters, including:
              - Gradient filters (nabla_x, nabla_y)
              - Gabor filters with varying sizes and orientations
              - Gaussian filters with varying sizes and sigmas
              - Laplacian filters with varying sizes

    Note:
        Ensure the filters are consistent in size during convolution, as filters
        with different shapes might require appropriate padding.
    """
    filters = []

    # Basic filter
    filters.append(np.array([[1]], dtype=np.float32))

    # Gradient filters (nabla_x and nabla_y)
    filters.extend([
        np.array([-1, 1], dtype=np.float32).reshape((1, 2)),  # nabla_x
        np.array([-1, 1], dtype=np.float32).reshape((2, 1))   # nabla_y
    ])

    # Gabor filters
    for size in [3, 5]:
        for theta in range(0, 180, 30):  # 0 to 150 degrees in steps of 30
            cosine, sine = gabor_filter(size, theta)
            filters.append(cosine)
            filters.append(sine)

    # Gaussian filters
    for size in [3, 5]:
        for sigma in range(1, 6, 2):  # Sigma values: 1, 3, 5
            filters.append(gaussian_filter(size, sigma))

    # Laplacian filters
    for size in [3, 5, 7]:
        filters.append(laplacian_filter(size))

    return filters


def LaplacianFilter(size):
    # 
    if size == 3:
        kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    elif size == 5:
        kernel = np.array([[0, 0, 1, 0, 0], [0, 1, 2, 1, 0], [
                          1, 2, -16, 2, 1], [0, 1, 2, 1, 0], [0, 0, 1, 0, 0]])
    elif size == 7:
        kernel = np.array([[0, 0, 0, 1, 0, 0, 0], [0, 0, 3, 2, 3, 0, 0], [0, 3, 2, 1, 2, 3, 0], [
                          1, 2, 1, -12, 1, 2, 1], [0, 3, 2, 1, 2, 3, 0], [0, 0, 3, 2, 3, 0, 0], [0, 0, 0, 1, 0, 0, 0]])
    return kernel



def gabor_filter(size, orientation):
    """
    Generates a Gabor filter (cosine and sine components) for a given size and orientation.

    Parameters:
        size (int): The size of the square filter (must be an odd positive integer).
        orientation (float): The orientation angle of the filter in degrees.

    Returns:
        tuple: A tuple containing two numpy arrays:
               - Cosine: The cosine component of the Gabor filter.
               - Sine: The sine component of the Gabor filter.
    """
    # Validate inputs
    if size % 2 == 0 or size <= 0:
        raise ValueError("Size must be a positive odd integer.")

    # Precompute constants
    halfsize = size // 2  # Integer division for the center index
    theta = np.deg2rad(orientation)  # Convert orientation to radians
    scale = size / 6  # Define the scale parameter

    # Create meshgrid for x and y coordinates
    y, x = np.meshgrid(np.arange(-halfsize, halfsize + 1), np.arange(-halfsize, halfsize + 1))

    # Rotate coordinates based on orientation
    x_rot = (x * np.cos(theta) + y * np.sin(theta)) / scale
    y_rot = (-x * np.sin(theta) + y * np.cos(theta)) / scale

    # Gaussian envelope
    gauss = np.exp(-0.5 * (x_rot**2 + 0.25 * y_rot**2))

    # Gabor filter components
    Cosine = gauss * np.cos(2 * x_rot)
    Sine = gauss * np.sin(2 * x_rot)

    # Normalize the cosine component
    normalization_factor = np.sum(Cosine) / np.sum(gauss)
    Cosine -= normalization_factor * gauss

    return Cosine.astype(np.float32), Sine.astype(np.float32)

# Example usage
cosine, sine = gabor_filter(size=15, orientation=45)
print("Cosine Component:\n", cosine)
print("Sine Component:\n", sine)
    
    return Cosine, Sine


def gaussianFilter(size, sigma):
    kernel = np.fromfunction(lambda x, y: (1/(2*np.pi*sigma**2)) * np.exp(-((x - size//2)**2 + (y - size//2)**2)/(2*sigma**2)), (size, size))
    return kernel / np.sum(kernel)






if __name__ == '__main__':
    F = get_filters()
    print(F)
