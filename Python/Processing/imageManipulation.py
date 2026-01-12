import numpy as np
import cv2
import plotly.express as px
from skimage import io, color, measure, filters, morphology

def morphological_effects(image, opening, closing, erosion, dilation, iter1, iter2, iter3, iter4, kernel_size):
    """
    Apply morphological operations to the input image.

    Args:
        image (numpy.ndarray): Input binary image.
        opening (bool): Flag to apply opening operation.
        closing (bool): Flag to apply closing operation.
        erosion (bool): Flag to apply erosion operation.
        dilation (bool): Flag to apply dilation operation.
        iter1 (int): Number of iterations for opening operation.
        iter2 (int): Number of iterations for closing operation.
        iter3 (int): Number of iterations for erosion operation.
        iter4 (int): Number of iterations for dilation operation.
        kernel_size (int): Size of the structuring element.

    Returns:
        numpy.ndarray: Image after applying morphological operations.
    """
    # Create kernel based on user input
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    # To prevent from overwriting image
    result = image.copy()

    # Apply erosion
    if erosion:
        result = cv2.erode(result, kernel, iterations=iter3)

    # Apply dilation
    if dilation:
        result = cv2.dilate(result, kernel, iterations=iter4)

    # Apply opening
    if opening:
        result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel, iterations=iter1)

    # Apply closing
    if closing:
        result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel, iterations=iter2)

    return result

def normalize_lighting(image):
    # Convert the image to grayscale (if it's not already)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

    gray_image = adjust_contrast(gray_image)

    # Apply GaussianBlur to remove noise and create a smoother version of the image
    blurred_image = cv2.GaussianBlur(gray_image, (21, 21), 0)

    # Subtract the blurred image from the original image (i.e., background subtraction)
    subtracted = cv2.subtract(gray_image, blurred_image)

    # Normalize the image intensity to enhance contrast
    normalized = cv2.normalize(subtracted, None, 0, 255, cv2.NORM_MINMAX)

    # Apply adaptive histogram equalization to enhance contrast locally (to mitigate uneven lighting)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized = clahe.apply(normalized)


    return equalized


def adjust_contrast(image, contrast_factor=3.0):
    """
    Adjust contrast of the image.

    Parameters:
    - image: Input image (grayscale or BGR).
    - contrast_factor: Factor to adjust contrast. Default is 1.0 (no change).
                       Values > 1.0 increase contrast, values < 1.0 decrease contrast.

    Returns:
    - The image with adjusted contrast.
    """
    # Optionally adjust contrast by scaling the intensity values
    adjusted = cv2.convertScaleAbs(image, alpha=contrast_factor, beta=0)
    return adjusted

def histogram_equalization(image):
    """
    Adjust edge brightness and apply histogram equalization to the input image.

    Args:
        image (numpy.ndarray): Input color image.

    Returns:
        numpy.ndarray: Grayscale image after histogram equalization.
    """
    height, width, _ = image.shape
    center_x, center_y = width / 2, height / 2

    # Calculate distances of each pixel from the image center
    y, x = np.ogrid[:height, :width]
    distance_to_center = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)

    # Calculate the maximum distance from the center
    max_distance = np.sqrt(center_x ** 2 + center_y ** 2)

    # Calculate brightness decrease factor based on distance
    brightness_decrease = distance_to_center / max_distance

    # Calculate the brightness change
    brightness_change = (brightness_decrease * 2).astype(np.uint8)

    # Split the image into channels
    b, g, r = cv2.split(image)

    # Add brightness_change to each channel separately
    b_adjusted = cv2.add(b, brightness_change)
    g_adjusted = cv2.add(g, brightness_change)
    r_adjusted = cv2.add(r, brightness_change)

    # Merge the adjusted channels back into an image
    image_with_brightness = cv2.merge((b_adjusted, g_adjusted, r_adjusted))

    # Convert to Grayscale
    image = cv2.cvtColor(image_with_brightness, cv2.COLOR_BGR2GRAY)

    # Apply histogram equalization
    equalized = cv2.equalizeHist(image)

    return equalized

def shadow_correction(image, block_size):
    """
    Apply automatic shadow correction by adjusting brightness of image sections.

    Args:
        image (numpy.ndarray): Input color image.
        block_size (int): Size of the blocks to divide the image.

    Returns:
        numpy.ndarray: Grayscale image after shadow correction.
    """
    def adjust_brightness(image):
        """
        Adjust brightness of each section of the image.

        Args:
            image (numpy.ndarray): Grayscale image.

        Returns:
            numpy.ndarray: Image with adjusted brightness.
        """
        height, width = image.shape[:2]

        # Calculate number of blocks in each dimension
        num_blocks_x = (width + block_size - 1) // block_size
        num_blocks_y = (height + block_size - 1) // block_size

        # Iterate over each block
        for i in range(num_blocks_y):
            for j in range(num_blocks_x):
                # Define block boundaries
                y_start = i * block_size
                y_end = min((i + 1) * block_size, height)
                x_start = j * block_size
                x_end = min((j + 1) * block_size, width)

                # Calculate average brightness of the block
                block = image[y_start:y_end, x_start:x_end]
                avg_brightness = np.mean(block)

                # Adjust brightness of the block
                image[y_start:y_end, x_start:x_end] = np.clip(block + (128 - avg_brightness), 0, 255)

        return image

    # Convert image to grayscale if it's not already
    if len(image.shape) > 2:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image

    # Apply histogram equalization
    equalized = cv2.equalizeHist(gray_image)

    # Adjust brightness of image sections
    corrected_image = adjust_brightness(equalized)

    return corrected_image

def manual_noise(image):
    """
    Apply manual noise reduction to the input image.

    Args:
        image (numpy.ndarray): Input binary image.

    Returns:
        numpy.ndarray: Image after manual noise reduction.
    """
    # Create a new image to store the results
    new_image = np.copy(image)

    # Define the 8 surrounding pixels
    neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    # Get the height and width of the image
    height, width = image.shape[:2]

    for y in range(height):
        for x in range(width):
            # Get the pixel value at (x, y)
            pixel = image[y, x]

            if pixel == 255:  # If the pixel is white
                # Count white neighbors
                white_neighbors = sum(
                    0 <= y + dy < height and 0 <= x + dx < width and image[y + dy, x + dx] == 255
                    for dx, dy in neighbors
                )
                if white_neighbors < 2:
                    new_image[y, x] = 0  # Change to black if fewer than 2 white neighbors
            elif pixel == 0:  # If the pixel is black
                # Check if all neighbors are white
                all_white_neighbors = all(
                    0 <= y + dy < height and 0 <= x + dx < width and image[y + dy, x + dx] == 255
                    for dx, dy in neighbors
                )
                if all_white_neighbors:
                    new_image[y, x] = 255  # Change to white if all neighbors are white

    return new_image

def median_filter(image):
    """
    Apply a median filter to remove noise from the input image.

    Args:
        image (numpy.ndarray): Input grayscale or binary image.

    Returns:
        numpy.ndarray: Image after applying the median filter.
    """
    # Apply a median filter to remove noise
    filtered_image = cv2.medianBlur(image, 3)  # 3x3 kernel size
    return filtered_image

def plot_fluorescent_histogram(average_intensities, bin_size=10):
    """
    Plot a histogram of average fluorescent intensities using Plotly.

    Args:
        average_intensities (list or numpy.ndarray): List of average fluorescent intensities.
        bin_size (int): Size of the bins in the histogram.

    Returns:
        plotly.graph_objects.Figure: Histogram plot of average fluorescent intensities.
    """
    # Create the histogram using Plotly
    fig = px.histogram(
        average_intensities,
        nbins=bin_size,
        title='Histogram of Average Fluorescent Intensities within Contours',
        labels={'value': 'Average Fluorescent Intensity', 'count': 'Frequency'},
    )

    # Customize the layout
    fig.update_layout(
        xaxis_title='Average Fluorescent Intensity',
        yaxis_title='Frequency',
        bargap=0.1,  # Gap between bars
    )

    return fig
