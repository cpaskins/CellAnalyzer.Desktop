import numpy as np
import cv2


# Various Edge Detection Algorithms
def sobel_filter(image):

    # Apply Gaussian blur to reduce noise and improve edge detection
    blurred_image = cv2.GaussianBlur(image, (5, 5), 1.4)

    # Apply Sobel filter in the x direction
    sobel_x = cv2.Sobel(blurred_image, cv2.CV_64F, dx=1, dy=0, ksize=1)

    # Apply Sobel filter in the y direction
    sobel_y = cv2.Sobel(image, cv2.CV_64F, dx=0, dy=1, ksize=1)

    # Compute the gradient magnitude
    gradient_magnitude = cv2.magnitude(sobel_x, sobel_y)

    # Normalize the result to fit in the range [0, 255]
    gradient_magnitude = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX)

    # Convert to uint8 type for displaying
    gradient_magnitude = gradient_magnitude.astype(np.uint8)

    # Convert to Grayscale
    if len(image.shape) > 2:
        gradient_magnitude = cv2.cvtColor(gradient_magnitude, cv2.COLOR_BGR2GRAY)

    equalized = cv2.equalizeHist(gradient_magnitude)

    return equalized


def canny_filter(image):
    # Apply Gaussian blur to reduce noise and improve edge detection
    blurred_image = cv2.GaussianBlur(image, (1, 1), 1.4)

    # Apply Canny edge detection
    edges = cv2.Canny(blurred_image, 0, 40)

    # Apply dilation to close gaps in edges
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)

    return dilated_edges

def laplace_filter(src):

    # Remove noise by blurring with a Gaussian filter
    src = cv2.GaussianBlur(src, (3, 3), 0)

    # Apply Laplace function
    dst = cv2.Laplacian(src, cv2.CV_16S, ksize=5)

    # converting back to uint8
    abs_dst = cv2.convertScaleAbs(dst)

    # Convert to Grayscale
    if len(abs_dst.shape) > 2:
        abs_dst = cv2.cvtColor(abs_dst, cv2.COLOR_BGR2GRAY)

    equalized = cv2.equalizeHist(abs_dst)

    return equalized


def prewitt_filter(image):
    # Convert to grayscale if the image is not already
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to the image
    img_gaussian = cv2.GaussianBlur(image, (3, 3), 0)

    # Define Prewitt kernels
    kernel_x = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
    kernel_y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])

    # Apply Prewitt kernels to the image
    prewitt_x = cv2.filter2D(img_gaussian, cv2.CV_64F, kernel_x)
    prewitt_y = cv2.filter2D(img_gaussian, cv2.CV_64F, kernel_y)

    # Combine the two results
    prewitt = np.sqrt(prewitt_x ** 2 + prewitt_y ** 2)

    # Normalize the result to range [0, 255]
    prewitt = cv2.normalize(prewitt, None, 0, 255, cv2.NORM_MINMAX)

    # Convert to uint8
    prewitt = prewitt.astype(np.uint8)

    # Equalize the histogram of the result
    equalized = cv2.equalizeHist(prewitt)

    return equalized


def roberts_cross_filter(image):
    # Convert to grayscale if the image is not already
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Define Roberts Cross kernels
    kernel_x = np.array([[1, 0], [0, -1]])
    kernel_y = np.array([[0, 1], [-1, 0]])

    # Apply Roberts Cross kernels to the image
    roberts_x = cv2.filter2D(image, cv2.CV_64F, kernel_x)
    roberts_y = cv2.filter2D(image, cv2.CV_64F, kernel_y)

    # Combine the two results
    roberts = np.sqrt(roberts_x ** 2 + roberts_y ** 2)

    # Normalize the result to range [0, 255]
    roberts = cv2.normalize(roberts, None, 0, 255, cv2.NORM_MINMAX)

    # Convert to uint8
    roberts = roberts.astype(np.uint8)

    # Equalize the histogram of the result
    equalized = cv2.equalizeHist(roberts)

    return equalized


def scharr_filter(image):
    # Convert to grayscale if the image is not already
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Scharr filter to the image
    scharr_x = cv2.Scharr(image, cv2.CV_64F, 1, 0)
    scharr_y = cv2.Scharr(image, cv2.CV_64F, 0, 1)

    # Combine the two results
    scharr = np.sqrt(scharr_x ** 2 + scharr_y ** 2)

    # Normalize the result to range [0, 255]
    scharr = cv2.normalize(scharr, None, 0, 255, cv2.NORM_MINMAX)

    # Convert to uint8
    scharr = scharr.astype(np.uint8)

    # Equalize the histogram of the result
    equalized = cv2.equalizeHist(scharr)

    return equalized


def frei_chen_filter(image):
    # Convert to grayscale if the image is not already
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Define Frei-Chen kernels
    kernel_x = np.array([[1, np.sqrt(2), 1], [0, 0, 0], [-1, -np.sqrt(2), -1]])
    kernel_y = np.array([[1, 0, -1], [np.sqrt(2), 0, -np.sqrt(2)], [1, 0, -1]])

    # Apply Frei-Chen kernels to the image
    frei_chen_x = cv2.filter2D(image, cv2.CV_64F, kernel_x)
    frei_chen_y = cv2.filter2D(image, cv2.CV_64F, kernel_y)

    # Combine the two results
    frei_chen = np.sqrt(frei_chen_x ** 2 + frei_chen_y ** 2)

    # Normalize the result to range [0, 255]
    frei_chen = cv2.normalize(frei_chen, None, 0, 255, cv2.NORM_MINMAX)

    # Convert to uint8
    frei_chen = frei_chen.astype(np.uint8)

    # Equalize the histogram of the result
    equalized = cv2.equalizeHist(frei_chen)

    return equalized

