import os
import imagehash
from PIL import Image

def delete_duplicate_images(folder_path, threshold=7):
    # Dictionary to store hashes of images
    image_hashes = {}

    # Traverse through the folder
    for filename in os.listdir(folder_path):
        file_path = os.path.normpath(os.path.abspath(os.path.join(folder_path, filename)))


        # Check if the file is an image
        if not os.path.isfile(file_path) or not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            continue
        
        try:
            # Open the image and calculate its hash
            with Image.open(file_path) as img:
                img_hash = imagehash.average_hash(img)

            # Check for duplicates with threshold
            duplicate_found = False
            for stored_hash, stored_path in image_hashes.items():
                if abs(img_hash - stored_hash) <= threshold:  # Compare hashes
                    print(f"Duplicate found: {file_path} is similar to {stored_path} -> Deleting...")
                    if os.access(file_path, os.W_OK):  # Check write permission
                        os.remove(file_path)
                        print(f"Deleted: {file_path}")
                    else:
                        print(f"Permission denied: {file_path}")  # Delete the duplicate image
                    duplicate_found = True
                    break
            
            if not duplicate_found:
                image_hashes[img_hash] = file_path

        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    print("Duplicate image deletion complete.")

# Specify the folder containing images
folder = "src/Ultimate/PT"
delete_duplicate_images(folder)
