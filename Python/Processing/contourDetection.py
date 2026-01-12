from skimage import io, color, measure, filters, morphology
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.ndimage import label, find_objects


def cv2_contours(original, gray, morphed, **kwargs):
    """
    Detect contours in a morphed image and process them to count cells, calculate cell areas,
    and optionally calculate fluorescence intensities.

    Args:
        original (numpy.ndarray): The original image.
        gray (numpy.ndarray): The grayscale version of the original image.
        morphed (numpy.ndarray): The morphed binary image for contour detection.
        **kwargs: Additional parameters for processing.

    Returns:
        tuple: The overlay image with drawn contours, number of cells, list of cell areas,
               and average intensities if fluorescence scoring is enabled.
    """
    fluorescence = kwargs['fluorescence']
    fluorescence_scoring = kwargs['fluorescence_scoring']
    minimum_area = kwargs['minimum_area']
    average_cell_area = kwargs['average_cell_area']
    connected_cell_area = kwargs['connected_cell_area']
    hole_size = kwargs['hole_size']
    hole_threshold = kwargs['hole_threshold']

    # Find contours in the morphed image
    cnts, hierarchy = cv2.findContours(morphed, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize variables for contour processing
    cells = 0
    cell_areas = []
    overlay = original.copy()  # Copy original image for drawing contours
    color = (36, 255, 12) if not fluorescence else (0, 100, 255)
    average_intensities = []

    # Process each contour
    for i, c in enumerate(cnts):
        # Only consider parent contours (outer contours)
        if hierarchy[0][i][3] == -1:  # No parent, it's an outer contour
            area = cv2.contourArea(c)

            # Calculate the area of holes within the contour
            holes_area = 0
            holes = []
            k = hierarchy[0][i][2]
            while k != -1:
                hole_area = cv2.contourArea(cnts[k])
                if hole_area > hole_size:
                    # Create a mask for the hole and calculate mean brightness
                    hole_mask = np.zeros(gray.shape, np.uint8)
                    cv2.drawContours(hole_mask, [cnts[k]], -1, (255), thickness=cv2.FILLED)
                    mean_brightness = cv2.mean(original, mask=hole_mask)[0]
                    if fluorescence:
                        mean_brightness = 255 - mean_brightness

                    # Filter out small holes based on brightness threshold
                    if mean_brightness < hole_threshold:
                        holes_area += hole_area
                        holes.append(cnts[k])
                k = hierarchy[0][k][0]

            # Subtract the total hole area from the outer contour area
            area -= holes_area

            # Only count contours larger than the minimum area
            if area > minimum_area:
                cv2.drawContours(overlay, [c], -1, color, 2)  # Draw outer contour
                for hole in holes:
                    cv2.drawContours(overlay, [hole], -1, (255, 0, 0), 2)  # Draw holes

                # Calculate the number of cells based on the area
                if area > connected_cell_area:
                    num_cells = math.ceil(area / average_cell_area)
                    cells += num_cells
                    cell_areas.extend([average_cell_area] * num_cells)
                else:
                    cells += 1
                    cell_areas.append(area)

                # Calculate average intensity within the contour if enabled
                if fluorescence_scoring:
                    contour_mask = np.zeros(gray.shape, np.uint8)
                    cv2.drawContours(contour_mask, [c], -1, (255), thickness=cv2.FILLED)
                    mean_intensity = cv2.mean(original, mask=contour_mask)[0]
                    average_intensities.append(mean_intensity)

    return overlay, cells, cell_areas, average_intensities


def scikit_contours(original, gray, morphed, **kwargs):
    """
    Detects contours in a grayscale image and processes them to count cells and calculate average intensities.

    Args:
        original (numpy.ndarray): The original image where contours will be drawn.
        gray (numpy.ndarray): The grayscale image.
        morphed (numpy.ndarray): The morphed image used for contour detection.
        min_area (int): Minimum area of contours to be detected.
        max_area (int): Maximum area of contours to be detected.

    Returns:
        tuple: A tuple containing the overlay image, number of cells, list of cell areas, and average intensities (if fluorescence_scoring is enabled).
    """
    # Find contours in the morphed image
    contours = measure.find_contours(morphed, level=0.5)

    # Initialize variables for contour processing
    cells = 0
    cell_areas = []
    average_intensities = []

    # Create an overlay image for drawing contours
    overlay = original.copy()
    overlay_color = (36, 255, 12) if not kwargs.get('fluorescence', True) else (0, 100, 255)

    # Process each contour
    for contour in contours:
        # Convert the contour to integer coordinates and switch x and y
        contour = np.array(contour, dtype=np.int32)
        contour = contour[:, [1, 0]]  # Switch x and y
        contour = contour.reshape((-1, 1, 2))

        # Compute the area of the contour
        contour_area = cv2.contourArea(np.array(contour, dtype=np.int32))

        if min_area <= contour_area <= max_area:
            # Draw the contour on the overlay image
            cv2.drawContours(overlay, [contour], -1, overlay_color, 2)

            # Count the number of cells based on the area
            cells += 1
            cell_areas.append(contour_area)

            # Calculate average intensity within the contour if enabled
            if kwargs.get('fluorescence_scoring', False):
                contour_mask = np.zeros(gray.shape, np.uint8)
                cv2.drawContours(contour_mask, [contour], -1, 255, thickness=cv2.FILLED)
                mean_intensity = cv2.mean(original, mask=contour_mask)[0]
                if kwargs.get('fluorescence', False):
                    mean_intensity = 255 - mean_intensity
                average_intensities.append(mean_intensity)

    return overlay, cells, cell_areas, average_intensities


def basic_contours(original, gray, morphed, **kwargs):
    """
    Detects contours by labeling connected components and processing them to count cells and calculate average intensities.

    Args:
        original (numpy.ndarray): The original image where contours will be drawn.
        gray (numpy.ndarray): The grayscale image.
        morphed (numpy.ndarray): The morphed binary image used for contour detection.
        min_area (int): Minimum area of contours to be detected.
        max_area (int): Maximum area of contours to be detected.

    Returns:
        tuple: A tuple containing the overlay image, number of cells, list of cell areas, and average intensities (if fluorescence_scoring is enabled).
    """
    min_area = kwargs['minimum_area']
    max_area = kwargs['connected_cell_area']

    # Label connected components in the morphed image
    labeled_image, num_features = label(morphed)

    # Initialize variables for contour processing
    cells = 0
    cell_areas = []
    average_intensities = []

    # Create an overlay image for drawing contours
    overlay = original.copy()
    overlay_color = (36, 255, 12) if not kwargs.get('fluorescence', True) else (0, 100, 255)

    # Process each labeled region
    for region_idx in range(1, num_features + 1):
        region_mask = (labeled_image == region_idx)
        area = np.sum(region_mask)

        if min_area <= area <= max_area:
            # Store the area
            cell_areas.append(area)
            cells += 1

            # Create a boundary mask (contour detection)
            boundary = np.zeros_like(morphed)
            boundary[region_mask] = 1
            contour_x, contour_y = np.gradient(boundary)
            contour = np.sqrt(contour_x ** 2 + contour_y ** 2) > 0

            # Draw the contour on the overlay image
            overlay[contour] = overlay_color

            # Calculate average intensity within the contour if enabled
            if kwargs.get('fluorescence_scoring', False):
                contour_mask = np.zeros(gray.shape, np.uint8)
                contour_mask[region_mask] = 255
                mean_intensity = cv2.mean(original, mask=contour_mask)[0]
                if kwargs.get('fluorescence', False):
                    mean_intensity = 255 - mean_intensity
                average_intensities.append(mean_intensity)

    return overlay, cells, cell_areas, average_intensities


def blobber(original, gray, morphed, min_threshold=100, max_threshold=255, min_area=1, max_area=50000,
            min_circularity=0.1, max_circularity=1.0, min_inertia_ratio=0.01, min_convexity=0.01, **kwargs):
    """
    Detects blobs in a grayscale image using the SimpleBlobDetector.

    Args:
        image (numpy.ndarray): The grayscale image where blobs will be detected.
        min_threshold (int): Minimum threshold for the blob detector.
        max_threshold (int): Maximum threshold for the blob detector.
        min_area (int): Minimum area of blobs to be detected.
        max_area (int): Maximum area of blobs to be detected.
        min_circularity (float): Minimum circularity of blobs to be detected.
        max_circularity (float): Maximum circularity of blobs to be detected.
        min_inertia_ratio (float): Minimum inertia ratio of blobs to be detected.
        min_convexity (float): Minimum convexity of blobs to be detected.

    Returns:
        list: A list of detected blob keypoints.
    """
    # Create a SimpleBlobDetector_Params object
    params = cv2.SimpleBlobDetector_Params()

    # Set the parameters for blob detection
    params.minThreshold = min_threshold
    params.maxThreshold = max_threshold
    params.filterByArea = False
    params.minArea = min_area
    params.maxArea = max_area
    params.filterByCircularity = False
    params.minCircularity = min_circularity
    params.maxCircularity = max_circularity
    params.filterByInertia = False
    params.minInertiaRatio = min_inertia_ratio
    params.filterByConvexity = False
    params.minConvexity = min_convexity

    # Create a SimpleBlobDetector with the specified parameters
    detector = cv2.SimpleBlobDetector_create(params)

    # Detect blobs in the image
    keypoints = detector.detect(morphed)

    fluorescence = kwargs['fluorescence']
    fluorescence_scoring = kwargs['fluorescence_scoring']
    minimum_area = kwargs['minimum_area']
    average_cell_area = kwargs['average_cell_area']
    connected_cell_area = kwargs['connected_cell_area']

    # Initialize variables for blob processing
    cells = 0
    cell_areas = []
    overlay = original.copy()  # Copy original image for drawing blobs
    color = (36, 255, 12) if not fluorescence else (0, 100, 255)
    average_intensities = []

    # Process each blob
    for kp in keypoints:
        x, y = kp.pt
        size = kp.size
        blob_mask = np.zeros(gray.shape, np.uint8)
        cv2.circle(blob_mask, (int(x), int(y)), int(size / 2), (255), thickness=cv2.FILLED)

        # Calculate the area of the blob
        area = np.sum(blob_mask) / 255  # Convert mask to area

        # Skip blobs that are smaller than the minimum area
        if area > minimum_area:
            # Draw the blob on the overlay
            cv2.circle(overlay, (int(x), int(y)), int(size / 2), color, 2)

            # Count the number of cells based on the area
            if area > connected_cell_area:
                num_cells = math.ceil(area / average_cell_area)
                cells += num_cells
                cell_areas.extend([average_cell_area] * num_cells)
            else:
                cells += 1
                cell_areas.append(area)

            # Calculate average intensity within the blob if enabled
            if fluorescence_scoring:
                mean_intensity = cv2.mean(original, mask=blob_mask)[0]
                if fluorescence:
                    mean_intensity = 255 - mean_intensity
                average_intensities.append(mean_intensity)

    return overlay, cells, cell_areas, average_intensities


def watershed_segmentation(image, morphed):
    """
    Applies the watershed algorithm for segmenting overlapping or confluent cells.

    Args:
        image (numpy.ndarray): The binary image to be segmented.
        morphed (numpy.ndarray): The morphed binary image used for preprocessing.

    Returns:
        image (numpy.ndarray): New segmented image
    """
    image = image.copy()

    # Perform morphological operations to remove small noises and fill gaps
    kernel = np.ones((3, 3), np.uint8)
    sure_bg = cv2.dilate(morphed, kernel, iterations=4)
    dist_transform = cv2.distanceTransform(morphed, cv2.DIST_L2, 3)
    _, sure_fg = cv2.threshold(dist_transform, 0.1 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    unknown = cv2.subtract(sure_bg, sure_fg)

    # Marker labelling
    _, markers = cv2.connectedComponents(sure_fg)

    # Add 1 to all the markers to distinguish sure regions
    markers = markers + 1

    # Mark the unknown regions with zero
    markers[unknown == 255] = 0

    # Apply the watershed algorithm
    markers = cv2.watershed(image, markers)
    image[markers == -1] = [255, 0, 0]  # Mark the boundaries in red

    # Convert markers to uint8 type
    segmented_image = np.uint8(markers)


    #segmented_image = cv2.bitwise_not(segmented_image)

    return segmented_image



def hough(image):
    """
    Detect circles in a grayscale image using the Hough Circle Transform.

    Args:
        image (numpy.ndarray): The grayscale image where circles will be detected.

    Returns:
        numpy.ndarray: Array of detected circles. Each circle is represented by (x, y, r) where
                       (x, y) is the center and r is the radius.
    """
    # Apply Hough Circle Transform to detect circles in the image
    circles = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30, param1=50, param2=30, minRadius=1,
                               maxRadius=30)
    return circles