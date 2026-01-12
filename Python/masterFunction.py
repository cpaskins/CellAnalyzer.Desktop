from Processing.edgeDetection import *
from Processing.contourDetection import *
from Processing.imageManipulation import *
from Processing.thresholding import *


def cell_detection(image, **kwargs):
    """
    Runs the full cell detection and analysis pipeline.

    This function is UI agnostic and is designed to be called from:
    - CLI interfaces
    - Desktop applications
    - Automated pipelines

    All visualization outputs are optional and returned as raw images.
    """

    params = kwargs
    normalized_check = True

    # Copy the original image to prevent overwriting
    original = image.copy()

    # Convert image to grayscale if it is not already
    try:
        gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    except Exception:
        gray = original

    #if fluorescence:
     #   gray = cv2.bitwise_not(gray)

    if normalized_check:
        normalized = normalize_lighting(gray)
    else:
        normalized = gray

    # Apply noise reduction if enabled
    if params['noise']:
        normalized = median_filter(normalized)

    # Selected edge detection method
    if params['image_method'] == "Sobel":
        processed = cv2.bitwise_not(sobel_filter(normalized))
    elif params['image_method'] == 'Canny':
        processed = cv2.bitwise_not(canny_filter(normalized))
    elif params['image_method'] == 'Laplace':
        processed = cv2.bitwise_not(laplace_filter(normalized))
    elif params['image_method'] == 'Prewitt':
        processed = cv2.bitwise_not(prewitt_filter(normalized))
    elif params['image_method'] == 'Roberts Cross':
        processed = cv2.bitwise_not(roberts_cross_filter(normalized))
    elif params['image_method'] == 'Scharr':
        processed = cv2.bitwise_not(scharr_filter(normalized))
    elif params['image_method'] == 'Frei Chen':
        processed = cv2.bitwise_not(frei_chen_filter(normalized))
    elif params['image_method'] == "Block Segmentation":
        processed = shadow_correction(normalized, params['block_size'])
    elif params['image_method'] == "Histogram":
        processed = histogram_equalization(normalized)
    elif params['image_method'] == 'Double':
        processed = cv2.bitwise_not(prewitt_filter(normalized))
        processed = cv2.bitwise_not(prewitt_filter(processed))
    else:
        processed = cv2.bitwise_not(gray.copy())

    # Apply Otsu's or global thresholding to create a binary mask
    mask = global_threshold(
        processed,
        params['lower_intensity'],
        params['upper_intensity']
    )

    # Apply morphological operations if enabled
    if params['morph_checkbox']:
        morphed = morphological_effects(
            mask.copy(),
            params['opening'],
            params['closing'],
            params['eroding'],
            params['dilating'],
            params['open_iter'],
            params['close_iter'],
            params['erode_iter'],
            params['dilate_iter'],
            params['kernel_size']
        )

    else:
        morphed = mask.copy()

    # Calculate the threshold area (number of non-zero pixels)
    threshold_area = cv2.countNonZero(morphed)


    # Detect contours and analyze cell areas and intensities
    overlay, cells, cell_areas, average_intensities = cv2_contours(original, gray, morphed, **kwargs)


    # Convert pixel area to real-world area using the scaling factor
    converted_area_total = int(sum(cell_areas) / params['scaling'] ** 2)
    converted_area_mean = round(np.mean(cell_areas) / params['scaling'] ** 2, 2) if cell_areas else 0
    converted_threshold_area = int(threshold_area / params['scaling'] ** 2)

    result = {
        "counts": {
            "cell_count": cells
        },
        "areas": {
            "total_contour_area": converted_area_total,
            "mean_contour_area": converted_area_mean,
            "total_threshold_area": converted_threshold_area
        },
        "fluorescence": {
            "average_intensities": average_intensities
        },
        "images": {
            "processed": processed,
            "mask": mask,
            "morphed": morphed,
            "overlay": overlay
        }
    }

    return result