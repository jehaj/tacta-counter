import cv2
import numpy as np
from collections import defaultdict


class CardPointCounter:
    def __init__(self):
        # Define HSV ranges for each color
        # Format: (lower_bound, upper_bound)
        self.color_ranges = {
            "green": (np.array([40, 40, 40]), np.array([80, 255, 255])),
            "red": [
                (np.array([0, 40, 40]), np.array([10, 255, 255])),
                (np.array([170, 40, 40]), np.array([180, 255, 255])),
            ],  # Red wraps around
            "yellow": (np.array([20, 40, 40]), np.array([40, 255, 255])),
            "blue": (np.array([90, 40, 40]), np.array([130, 255, 255])),
            "pink": (np.array([140, 40, 40]), np.array([170, 255, 255])),
        }

        # White dot detection parameters (HSV)
        self.white_lower = np.array([0, 0, 180])
        self.white_upper = np.array([180, 30, 255])

        # White purity threshold - what % of pixels must be white
        self.white_purity_threshold = 0.95  # Easy to change: 0.95 = 95%

    def is_white_dot(self, hsv_image, contour):
        """Check if a contour is actually a white dot by checking pixel purity."""
        # Create a mask for just this contour
        contour_mask = np.zeros(hsv_image.shape[:2], dtype=np.uint8)
        cv2.drawContours(contour_mask, [contour], -1, 255, -1)

        # Count pixels in the contour
        total_pixels = cv2.countNonZero(contour_mask)
        if total_pixels == 0:
            return False

        # Check each pixel in the contour against white range
        white_check_mask = cv2.inRange(hsv_image, self.white_lower, self.white_upper)

        # Count white pixels within the contour
        white_pixels = cv2.countNonZero(cv2.bitwise_and(white_check_mask, contour_mask))

        # Calculate purity
        purity = white_pixels / total_pixels

        return purity >= self.white_purity_threshold

    def count_points(self, image_path):
        """Main function to count points from an image."""
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")

        # Convert to HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Detect white dots
        white_mask = cv2.inRange(hsv, self.white_lower, self.white_upper)

        # Find white dot contours
        contours, _ = cv2.findContours(
            white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Get centroids of white dots
        dot_centers = []
        for cnt in contours:
            # Filter by area to avoid noise
            area = cv2.contourArea(cnt)
            if area > 10:  # Adjust threshold as needed
                # Check if it's actually a white dot
                if self.is_white_dot(hsv, cnt):
                    M = cv2.moments(cnt)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        dot_centers.append((cx, cy))

        # Count dots by color
        points = defaultdict(int)

        for color, ranges in self.color_ranges.items():
            # Create mask for this color
            if isinstance(ranges, list):  # Handle red's wrap-around
                mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
                for lower, upper in ranges:
                    mask |= cv2.inRange(hsv, lower, upper)
            else:
                lower, upper = ranges
                mask = cv2.inRange(hsv, lower, upper)

            # Dilate to connect nearby regions
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=2)

            # Check which dots fall in this color region
            for cx, cy in dot_centers:
                if mask[cy, cx] > 0:
                    points[color] += 1

        return points

    def count_points_with_visualization(self, image_path, output_path=None):
        """Count points and create visualization."""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Detect white dots
        white_mask = cv2.inRange(hsv, self.white_lower, self.white_upper)
        contours, _ = cv2.findContours(
            white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Visualization image
        vis_img = img.copy()

        # Color mapping for visualization
        color_bgr = {
            "green": (0, 255, 0),
            "red": (0, 0, 255),
            "yellow": (0, 255, 255),
            "blue": (255, 0, 0),
            "pink": (255, 0, 255),
        }

        dot_centers = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 10:
                # Check if it's actually a white dot
                if self.is_white_dot(hsv, cnt):
                    M = cv2.moments(cnt)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        dot_centers.append((cx, cy))

        points = defaultdict(int)

        for color, ranges in self.color_ranges.items():
            if isinstance(ranges, list):
                mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
                for lower, upper in ranges:
                    mask |= cv2.inRange(hsv, lower, upper)
            else:
                lower, upper = ranges
                mask = cv2.inRange(hsv, lower, upper)

            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=2)

            # Check and draw dots
            for cx, cy in dot_centers:
                if mask[cy, cx] > 0:
                    points[color] += 1
                    cv2.circle(vis_img, (cx, cy), 8, color_bgr[color], 2)

        # Add text overlay with scores
        y_offset = 30
        for color, count in sorted(points.items()):
            text = f"{color.capitalize()}: {count}"
            cv2.putText(
                vis_img,
                text,
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
            y_offset += 30

        if output_path:
            cv2.imwrite(output_path, vis_img)

        return points, vis_img


# Example usage
if __name__ == "__main__":
    counter = CardPointCounter()

    # Basic counting
    try:
        points = counter.count_points("data/20251106_201644.jpg")
        print("Points by color:")
        for color, count in sorted(points.items()):
            print(f"  {color.capitalize()}: {count}")
        print(f"\nTotal: {sum(points.values())} points")
    except Exception as e:
        print(f"Error: {e}")

    # With visualization
    try:
        points, vis = counter.count_points_with_visualization(
            "data/20251106_201644.jpg", "data/result.jpg"
        )
        print("\nVisualization saved to data/result.jpg")
    except Exception as e:
        print(f"Error: {e}")
