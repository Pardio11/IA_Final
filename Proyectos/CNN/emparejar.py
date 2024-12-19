import os
import random

def remove_files_till_count(directory, target_count):
    """Remove files from the directory randomly until the number of files is <= target_count."""
    # Get a list of all files
    files = [os.path.join(directory, f) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

    # Shuffle files to randomize deletion
    random.shuffle(files)

    while len(files) > target_count:
        file_to_remove = files.pop()
        print(f"Removing: {file_to_remove}")
        os.remove(file_to_remove)

    print(f"Final number of files: {len(files)}")

if __name__ == "__main__":
    dir_path = input("Enter the directory path: ").strip()
    target_count = int(input("Enter the target number of files: "))

    if not os.path.isdir(dir_path):
        print("Invalid directory path.")
    else:
        remove_files_till_count(dir_path, target_count)
