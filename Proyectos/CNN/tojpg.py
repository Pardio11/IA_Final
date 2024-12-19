from PIL import Image
import os

def convert_images_in_directory(directory):
    """
    Convert all images in the given directory to JPG format, replacing the originals.

    Parameters:
        directory (str): Path to the directory containing images.
    """
    # Resolve the absolute path of the directory
    abs_directory = os.path.abspath(directory)

    if not os.path.isdir(abs_directory):
        print(f"Error: {abs_directory} is not a valid directory.")
        return

    # Iterate through all files in the directory
    for filename in os.listdir(abs_directory):
        input_path = os.path.join(abs_directory, filename)

        # Skip non-files
        if not os.path.isfile(input_path):
            continue

        # Skip already JPG files
        if filename.lower().endswith(".jpg"):
            print(f"Skipping {filename} (already a JPG).")
            continue

        # Convert the file
        try:
            with Image.open(input_path) as img:
                # Create output path
                base_name, _ = os.path.splitext(filename)
                output_path = os.path.join(abs_directory, f"{base_name}.jpg")

                # Convert to JPG
                img.convert("RGB").save(output_path, "JPEG")
                print(f"Converted {filename} to {output_path}")

                # Remove the original file if different
                if input_path != output_path:
                    os.remove(input_path)
        except Exception as e:
            print(f"Error converting {filename}: {e}")

# Example usage
if __name__ == "__main__":
    directory = input("Enter the relative path to the directory: ")
    convert_images_in_directory(directory)
