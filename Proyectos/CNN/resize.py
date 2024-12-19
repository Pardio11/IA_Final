from PIL import Image
import os

def resize_images_in_directory(input_dir, size=(40, 40)):
    """
    Resize all images in a directory to the specified size and overwrite the originals.

    Args:
        input_dir (str): Path to the directory containing images.
        size (tuple): Target size as (width, height).
    """
    for filename in os.listdir(input_dir):
        filepath = os.path.join(input_dir, filename)
        
        try:
            with Image.open(filepath) as img:
                img_resized = img.resize(size, Image.ANTIALIAS)  # Resize image
                img_resized.save(filepath)  # Overwrite the original image
                print(f"Resized and saved: {filepath}")
        except Exception as e:
            print(f"Skipping file {filename}. Error: {e}")

# Example usage
input_directory = "src/Ultimate/Combi"
resize_size = (40, 40)  # Desired width and height

resize_images_in_directory(input_directory, resize_size)