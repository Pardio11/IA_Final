import os

# Path to the file you want to delete
file_path = "src/Ultimate/PT/Combi2_27253.jpg"

# Check if the file exists
if os.path.exists(file_path):
    try:
        os.remove(file_path)
        print(f"File deleted successfully: {file_path}")
    except Exception as e:
        print(f"Error deleting file: {e}")
else:
    print(f"File not found: {file_path}")
