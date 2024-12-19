import os
import cv2  # OpenCV library
import numpy as np

def calculate_sharpness(image_path):
    """Calculate the sharpness of an image using Laplacian variance."""
    try:
        # Read the image in grayscale
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"Error: Cannot read image {image_path}")
            return 0

        # Compute the Laplacian of the image and calculate the variance
        laplacian = cv2.Laplacian(image, cv2.CV_64F)
        variance = laplacian.var()

        return variance
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return 0

def delete_blurry_images(folder_path, threshold=100.0):
    """
    Delete images that are considered blurry based on a sharpness threshold.
    :param folder_path: Path to the folder containing images.
    :param threshold: Laplacian variance threshold below which images are considered blurry.
    """
    for filename in os.listdir(folder_path):
        file_path = os.path.abspath(os.path.join(folder_path, filename)).replace("\\", "/")

        if 'src/' in file_path:
            file_path = file_path[file_path.index('src/'):]
        # Check if the file is an image
        if not os.path.isfile(file_path) or not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            continue

        # Calculate sharpness
        sharpness = calculate_sharpness(file_path)
        print(f"Image: {file_path} | Sharpness: {sharpness:.2f}")

        # Delete if the image is blurry
        if sharpness < threshold:
            try:
                os.remove(file_path)
                print(f"Deleted blurry image: {file_path}")
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")

    print("Blurry image deletion complete.")

# Specify the folder containing images
folder = "src/Ultimate/PT"
sharpness_threshold = 100.0  # Adjust threshold as needed
delete_blurry_images(folder, sharpness_threshold)