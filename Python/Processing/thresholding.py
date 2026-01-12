import cv2
import numpy as np

def adaptive_threshold(image, lower_thresh, upper_thresh):
    """
    Apply adaptive thresholding to the input image.

    Args:
        image (numpy.ndarray): Input grayscale image.
        lower_thresh (int): Lower threshold value (unused in this function).
        upper_thresh (int): Upper threshold value for binary thresholding.

    Returns:
        numpy.ndarray: Binary mask obtained after adaptive thresholding.
    """
    # Apply Gaussian blur to the image
    blur = cv2.GaussianBlur(image, (5, 5), 0)
    # Apply adaptive thresholding using Gaussian method
    mask = cv2.adaptiveThreshold(blur, upper_thresh, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    return mask


def otsu_threshold(image, lower_thresh, upper_thresh):
    """
    Apply Otsu's thresholding to the input image.

    Args:
        image (numpy.ndarray): Input grayscale image.
        lower_thresh (int): Lower threshold value.
        upper_thresh (int): Upper threshold value for binary thresholding.

    Returns:
        numpy.ndarray: Binary mask obtained after Otsu's thresholding.
    """
    # Apply Gaussian blur to the image
    blur = cv2.GaussianBlur(image, (5, 5), 0)
    # Apply Otsu's thresholding
    _, mask = cv2.threshold(blur, lower_thresh, upper_thresh, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    return mask


def global_threshold(image, lower_thresh, upper_thresh):
    """
    Apply global thresholding to the input image.

    Args:
        image (numpy.ndarray): Input grayscale image.
        lower_thresh (int): Lower threshold value.
        upper_thresh (int): Upper threshold value for binary thresholding (unused in this function).

    Returns:
        numpy.ndarray: Binary mask obtained after global thresholding.
    """
    # Apply global thresholding with a fixed threshold value of 127
    _, mask = cv2.threshold(image, lower_thresh, upper_thresh, cv2.THRESH_BINARY_INV)

    return mask
