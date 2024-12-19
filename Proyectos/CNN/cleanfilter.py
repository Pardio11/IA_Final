import os

def delete_files_with_colors(folder_path):
    """
    Deletes files in the specified folder that contain 'red', 'green', or 'blue' in their names.

    :param folder_path: Path to the folder to process
    """
    if not os.path.isdir(folder_path):
        print(f"Error: {folder_path} is not a valid directory.")
        return

    colors = ['green', 'red', 'blue','variant']

    # Iterate through all files in the folder
    for filename in os.listdir(folder_path):
        # Check if the file contains any of the specified colors
        if any(color in filename.lower() for color in colors):
            file_path = os.path.join(folder_path, filename)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    print(f"Deleted: {file_path}")
                else:
                    print(f"Skipped (not a file): {file_path}")
            except Exception as e:
                print(f"Failed to delete {file_path}: {e}")

if __name__ == "__main__":
    folder_path = input("Enter the path to the folder: ").strip()
    delete_files_with_colors(folder_path)
