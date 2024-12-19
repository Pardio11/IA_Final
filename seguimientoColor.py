import cv2
import numpy as np
import time
import os

# Ensure the save directory exists
save_dir = 'imgs/object/'
os.makedirs(save_dir, exist_ok=True)

# Start video capture
cap = cv2.VideoCapture(0)

# Define color ranges in HSV
lower_blue = np.array([80, 150, 0])
upper_blue = np.array([100, 255, 255])

lower_black = np.array([0, 0, 0]) 
upper_black = np.array([180, 255, 65])

count = 0

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to capture video.")
        break
    
    # Convert the frame to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Create masks for blue and black
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    mask_blue = cv2.erode(mask_blue, None, iterations=2)
    mask_blue = cv2.dilate(mask_blue, None, iterations=2)

    mask_black = cv2.inRange(hsv, lower_black, upper_black)
    mask_black = cv2.erode(mask_black, None, iterations=2)
    mask_black = cv2.dilate(mask_black, None, iterations=2)

    # Combine masks
    mask_combined = mask_blue + mask_black

    # Find contours in the combined mask
    contours, _ = cv2.findContours(mask_combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Process the largest contour if it exists
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(largest_contour)
        
        # Draw and process if the radius is large enough
        if radius > 10:
            top_left = (max(0, int(x - radius)), max(0, int(y - radius)))
            bottom_right = (min(frame.shape[1], int(x + radius)), min(frame.shape[0], int(y + radius)))

            if radius >= 50:
                # Extract the region of interest (ROI)
                img2 = frame[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

                if img2.size > 0:  # Ensure ROI is valid
                    # Resize the image
                    img2 = cv2.resize(img2, (100, 100))

                    # Display and save the image
                    cv2.imshow('Extracted Image', img2)
                    count += 1
                    filename = os.path.join(save_dir, f'mouse{count}.jpg')
                    cv2.imwrite(filename, img2)
                    print(f"Saved: {filename}")

                    # Wait briefly
                    time.sleep(0.5)
    
    # Show the frames
    cv2.imshow('Frame', frame)
    cv2.imshow('Mask Combined', mask_combined)

    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
