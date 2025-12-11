import math
import cv2
import numpy as np
from numpy.typing import NDArray
from typing import Any, TypeAlias

HSVCartesian: TypeAlias = tuple[float, float, float]
HSVColor: TypeAlias = tuple[int, int, int]
RGBColor: TypeAlias = tuple[int, int, int]


def load_image(image_path: str) -> tuple[NDArray, NDArray, NDArray]:
    """Load image and return RGB, grayscale, and HSV versions.

    Args:
        image_path: Path to the image file

    Returns:
        Tuple of (image_rgb, gray, image_hsv)

    Raises:
        FileNotFoundError: If image file is not found
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    image_hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)

    return image_rgb, gray, image_hsv


def apply_adaptive_threshold(
    gray: NDArray,
    blur_kernel: tuple[int, int] = (11, 11),
    block_size: int = 21,
    c: int = -5,
) -> NDArray:
    """Apply Gaussian blur and adaptive thresholding to detect edges.

    Args:
        gray: Grayscale image
        blur_kernel: Gaussian blur kernel size (must be odd numbers)
        block_size: Size of pixel neighborhood for threshold calculation (must be odd)
        c: Constant subtracted from weighted mean

    Returns:
        Binary thresholded image
    """
    blurred = cv2.GaussianBlur(gray, blur_kernel, 0)
    adaptive_thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, c
    )
    return adaptive_thresh


def find_contours(adaptive_thresh: NDArray) -> tuple:
    """Find contours in thresholded image.

    Args:
        adaptive_thresh: Binary thresholded image

    Returns:
        Tuple of (contours, hierarchy)
    """
    return cv2.findContours(adaptive_thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)


def darken_image(image_rgb: NDArray, factor: float = 0.6) -> NDArray:
    """Darken an image by a given factor.

    Args:
        image_rgb: RGB image
        factor: Darkening factor (0-1, where 1 is no change)

    Returns:
        Darkened image
    """
    return (image_rgb * factor).astype(np.uint8)


def filter_circular_contours(
    contours: tuple,
    gray: NDArray,
    image_hsv: NDArray,
    min_diameter: int = 8,
    max_diameter: int = 18,
    min_intensity: float = 100.0,
    max_std_dev: float = 20.0,
    max_saturation: float = 80.0,
    min_circularity: float = 0.4,
    min_area: float = 30.0,
    min_perimeter: float = 10.0,
) -> list[NDArray]:
    """Filter contours to find circles with white/gray fill and low color saturation.

    Args:
        contours: All detected contours
        gray: Grayscale image
        image_hsv: HSV image
        min_diameter: Minimum circle diameter in pixels
        max_diameter: Maximum circle diameter in pixels
        min_intensity: Minimum mean intensity (0-255)
        max_std_dev: Maximum standard deviation of intensity
        max_saturation: Maximum mean saturation (0-255)
        min_circularity: Minimum circularity (1.0 is perfect circle)
        min_area: Minimum contour area
        min_perimeter: Minimum contour perimeter

    Returns:
        List of filtered contours matching criteria
    """
    filtered_contours = []
    min_radius = min_diameter / 2
    max_radius = max_diameter / 2

    for contour in contours:
        # Calculate contour properties
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)

        if area < min_area or perimeter <= min_perimeter:
            continue

        # Calculate circularity
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter * perimeter)

        if circularity <= min_circularity:
            continue

        # Check radius
        (x, y), radius = cv2.minEnclosingCircle(contour)
        if not (min_radius <= radius <= max_radius):
            continue

        # Create mask for the circle region
        mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.circle(mask, (int(x), int(y)), int(radius) - 2, 255, -1)

        # Check intensity (white/gray fill)
        pixels = gray[mask == 255]
        mean_intensity = pixels.mean()
        std_intensity = pixels.std()

        if mean_intensity <= min_intensity or std_intensity >= max_std_dev:
            continue

        # Check saturation (low color)
        hsv_masked = cv2.bitwise_and(image_hsv, image_hsv, mask=mask)
        saturation_values = hsv_masked[:, :, 1][mask == 255]
        mean_saturation = np.mean(saturation_values)

        if mean_saturation >= max_saturation:
            continue

        filtered_contours.append(contour)

    return filtered_contours


def extract_donut_colors(
    contours: list[NDArray],
    image_rgb: NDArray,
    inner_offset: int = 2,
    outer_offset: int = 6,
) -> list[HSVColor]:
    """Extract colors from donut regions around circles.

    Args:
        contours: List of circular contours
        image_rgb: RGB image
        inner_offset: Inner radius offset from circle edge
        outer_offset: Outer radius offset from circle edge

    Returns:
        List of HSV color tuples for each contour
    """
    hsv_colors = []
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

    for contour in contours:
        (x, y), radius = cv2.minEnclosingCircle(contour)

        # Create donut mask
        circle_inner = cv2.circle(
            np.zeros_like(gray), (int(x), int(y)), int(radius) + inner_offset, 255, -1
        )
        circle_outer = cv2.circle(
            np.zeros_like(gray), (int(x), int(y)), int(radius) + outer_offset, 255, -1
        )
        donut = circle_outer - circle_inner

        # Get mean color in donut region
        scalar = cv2.mean(image_rgb, mask=donut)
        r, g, b, _ = scalar  # type: ignore

        # Convert RGB to HSV
        rgb_array = np.uint8([[[int(r), int(g), int(b)]]])  # type: ignore
        hsv_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2HSV)  # type: ignore
        h, s, v = hsv_array[0][0]  # type: ignore

        hsv_colors.append((int(h), int(s), int(v)))

    return hsv_colors


def convert_hsv_to_cartesian_array(
    hsv_colors: list[HSVColor],
) -> tuple[list[float], list[float], list[float]]:
    """Convert list of HSV colors to Cartesian coordinates for 3D plotting.

    Args:
        hsv_colors: List of HSV color tuples

    Returns:
        Tuple of (x_coords, y_coords, z_coords)
    """
    x_coords = []
    y_coords = []
    z_coords = []

    for h, s, v in hsv_colors:
        angle, radius, height = hsv_to_cartesian(h, s, v)
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        z = height

        x_coords.append(x)
        y_coords.append(y)
        z_coords.append(z)

    return x_coords, y_coords, z_coords


def hsv_to_rgb_string(h: int, s: int, v: int) -> str:
    """Convert HSV color to RGB string for plotting.

    Args:
        h: Hue (0-179)
        s: Saturation (0-255)
        v: Value (0-255)

    Returns:
        RGB string in format "rgb(r, g, b)"
    """
    hsv_array = np.uint8([[[h, s, v]]])  # type: ignore
    rgb_array = cv2.cvtColor(hsv_array, cv2.COLOR_HSV2RGB)  # type: ignore
    rgb = rgb_array[0][0]  # type: ignore
    return f"rgb({rgb[0]}, {rgb[1]}, {rgb[2]})"


def assign_clusters_to_colors(
    cluster_centers: NDArray, reference_colors: dict[str, HSVColor]
) -> dict[int, str]:
    """Assign each cluster to its closest reference color using greedy matching.

    Args:
        cluster_centers: K-means cluster centers in HSV space
        reference_colors: Dictionary mapping color names to HSV tuples

    Returns:
        Dictionary mapping cluster IDs to color names
    """
    cluster_to_color = {}
    used_colors = set()

    # Create list of (cluster_id, color_name, distance) for all combinations
    assignments = []
    for i, center in enumerate(cluster_centers):
        current = np.array(center)

        for color_name, ref_hsv in reference_colors.items():
            ref = np.array(ref_hsv)
            distance = np.linalg.norm(current - ref)
            assignments.append((i, color_name, distance))

    # Sort by distance and assign greedily
    assignments.sort(key=lambda x: x[2])

    for cluster_id, color_name, distance in assignments:
        if cluster_id not in cluster_to_color and color_name not in used_colors:
            cluster_to_color[cluster_id] = color_name
            used_colors.add(color_name)

    return cluster_to_color


def hsv_to_cartesian(h: int, s: int, v: int) -> HSVCartesian:
    """Convert HSV color to Cartesian coordinates.

    HSV should have values in the range:
    - h: 0-179
    - s: 0-255
    - v: 0-255

    Outputs x, y, z coordinates in the range:
    - x: 0 to 2 pi radians
    - y: 0 to 1
    - z: 0 to 1
    """
    # Convert hue from [0, 179] to [0, 2*pi]
    x = (h / 180) * 2 * math.pi

    # Convert saturation from [0, 255] to [0, 1]
    y = s / 255

    # Convert value from [0, 255] to [0, 1]
    z = v / 255

    return x, y, z


def cartesian_to_hsv(x: float, y: float, z: float) -> tuple[int, int, int]:
    """Convert Cartesian coordinates back to HSV color.

    Inputs x, y, z coordinates in the range:
    - x: 0 to 2 pi radians
    - y: 0 to 1
    - z: 0 to 1

    Outputs HSV should have values in the range:
    - h: 0-179
    - s: 0-255
    - v: 0-255
    """
    # Convert x from [0, 2*pi] to [0, 179]
    h = int((x / (2 * math.pi)) * 180) % 180

    # Convert y from [0, 1] to [0, 255]
    s = int(y * 255)

    # Convert z from [0, 1] to [0, 255]
    v = int(z * 255)

    return h, s, v
