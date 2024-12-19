import cv2
import numpy as np
from collections import deque

def is_color_in_range(hsv_pixel, lower_bound, upper_bound):
    """Checks if an HSV pixel is within a specified range."""
    return cv2.inRange(hsv_pixel, lower_bound, upper_bound) > 0

def perform_flood_fill(hsv, start_point, lower_bound, upper_bound, visited, min_size):
    """Flood-fill algorithm to detect connected regions of a specified color."""
    height, width = hsv.shape[:2]
    object_pixels = []
    queue = deque([start_point])

    while queue:
        x, y = queue.popleft()

        if visited[y, x]:
            continue

        visited[y, x] = True
        current_color = hsv[y, x]

        if is_color_in_range(current_color, lower_bound, upper_bound):
            object_pixels.append((x, y))

            # Check neighboring pixels
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < width and 0 <= ny < height and not visited[ny, nx]:
                    queue.append((nx, ny))

    if len(object_pixels) >= min_size:
        return object_pixels
    return []

def detect_red_objects(image, lower_bound, upper_bound, min_size):
    """Detects red objects in an image based on HSV color ranges."""
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    visited = np.zeros(hsv_image.shape[:2], dtype=bool)
    detected_centers = []

    for y in range(hsv_image.shape[0]):
        for x in range(hsv_image.shape[1]):
            if not visited[y, x] and is_color_in_range(hsv_image[y, x], lower_bound, upper_bound):
                object_pixels = perform_flood_fill(hsv_image, (x, y), lower_bound, upper_bound, visited, min_size)
                if object_pixels:
                    center = np.mean(object_pixels, axis=0)
                    detected_centers.append((int(center[0]), int(center[1])))

    return detected_centers

def main():
    # Define the HSV range for red color
    lower_red1 = np.array([0, 150, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 150, 50])
    upper_red2 = np.array([180, 255, 255])

    # Minimum size for objects to be considered
    min_size = 50

    # Load the image
    image_path = 'salida.jpg'
    image = cv2.imread(image_path)

    # Detect red objects in both HSV ranges
    red_objects = detect_red_objects(image, lower_red1, upper_red1, min_size)
    red_objects.extend(detect_red_objects(image, lower_red2, upper_red2, min_size))

    # Print and mark the detected object centers
    for idx, center in enumerate(red_objects):
        print(f"Red object {idx + 1} center: {center}")
        cv2.circle(image, center=center, radius=5, color=(0, 255, 255), thickness=2)

    # Save and display the result
    output_image_path = 'detected_red_objects.jpg'
    cv2.imwrite(output_image_path, image)
    cv2.imshow('Red Objects Detection', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()