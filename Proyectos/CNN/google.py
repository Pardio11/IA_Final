import cv2
import os
import re
import numpy as np
from google_images_search import GoogleImagesSearch

def process_and_detect_cars(query_string, output_dir, yolo_weights, yolo_cfg, coco_names, limit=5, image_size=(80, 80), api_key="", cse_id=""):

    os.makedirs(output_dir, exist_ok=True)

    # Get the last index based on existing files
    existing_files = [f for f in os.listdir(output_dir) if f.startswith("Savana") and f.endswith(".jpg")]
    if existing_files:
        last_index = max([int(re.search(r"Savana_(\d+)", f).group(1)) for f in existing_files])
    else:
        last_index = -1  # No existing files

    frame_count = last_index + 1

    # Configure Google Image Search
    gis = GoogleImagesSearch(api_key, cse_id)

    # Search and download images
    search_params = {
        'q': query_string,
        'num': limit,
        'fileType': 'jpg',
        'safe': 'high',
        'imgType': 'photo',
    }
    downloaded_dir = os.path.join("temp", query_string)
    os.makedirs(downloaded_dir, exist_ok=True)

    gis.search(search_params=search_params, path_to_dir=downloaded_dir)

    # Configure YOLO
    net = cv2.dnn.readNet(yolo_weights, yolo_cfg)
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    # Load the classes from the model
    with open(coco_names, "r") as f:
        classes = f.read().strip().split("\n")

    # Process each downloaded image
    for file_name in os.listdir(downloaded_dir):
        image_path = os.path.join(downloaded_dir, file_name)
        image = cv2.imread(image_path)
        if image is None:
            continue

        # Display the image on the screen
        cv2.imshow("Image", image)
        key = cv2.waitKey(0)  # Wait for a key press

        if key == 13:  # Enter key
            print("Processing image...")
            height, width, _ = image.shape
            blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
            net.setInput(blob)
            detections = net.forward(output_layers)

            for detection in detections:
                for obj in detection:
                    scores = obj[5:]
                    class_id = int(scores.argmax())
                    confidence = scores[class_id]

                    if classes[class_id] == "car" and confidence > 0.5:
                        # Get bounding box coordinates
                        center_x, center_y, w, h = (obj[0:4] * [width, height, width, height]).astype("int")
                        x = max(0, int(center_x - w / 2))
                        y = max(0, int(center_y - h / 2))
                        w = min(w, width - x)
                        h = min(h, height - y)

                        # Crop the car image
                        cropped_car = image[y:y+h, x:x+w]
                        resized_car = cv2.resize(cropped_car, image_size)

                        # Save the image with rotations
                        for angle in range(0, 360, 60):
                            M = cv2.getRotationMatrix2D((image_size[0] // 2, image_size[1] // 2), angle, 1.0)
                            rotated_car = cv2.warpAffine(resized_car, M, image_size)

                            frame_name = os.path.join(output_dir, f"Savana_{frame_count:05d}.jpg")
                            cv2.imwrite(frame_name, rotated_car)
                            print(f"Saved {frame_name}")
                            frame_count += 1
        elif key == 32:  # Spacebar key
            print("Skipping image...")

        # Close the image window
        cv2.destroyWindow("Image")

    # Cleanup temporary directory
    for file_name in os.listdir(downloaded_dir):
        os.remove(os.path.join(downloaded_dir, file_name))
    os.rmdir(downloaded_dir)
    print(f"Processed and saved {frame_count - (last_index + 1)} car images to {output_dir}")

# Example usage
process_and_detect_cars(
    query_string="GMC Savana Van",
    output_dir="src/dataset_2/Savana/",
    yolo_weights="src/yolov3.weights", 
    yolo_cfg="src/yolov3.cfg",        
    coco_names="src/coco.names",     
    limit=80,                           
    image_size=(80, 80),               
    api_key="YOUR_GOOGLE_API_KEY",    
    cse_id="YOUR_GOOGLE_CSE_ID"       
)
