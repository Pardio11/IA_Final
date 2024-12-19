import numpy as np
import cv2 as cv

# Load the face classifier
face_cascade = cv.CascadeClassifier('haarcascade_frontalface_alt.xml')

# Start video capture
video_capture = cv.VideoCapture(0)

def process_frame(frame):
    """Processes a single frame to detect faces and analyze pixel data."""
    # Convert the frame to grayscale
    grayscale_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(grayscale_frame, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Draw a rectangle around the detected face
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Extract the face region
        face_region = frame[y:y + h, x:x + w]

        # Resize the face region to 80x80 pixels
        resized_face = cv.resize(face_region, (80, 80), interpolation=cv.INTER_AREA)

        # Convert the resized face to grayscale
        gray_face = cv.cvtColor(resized_face, cv.COLOR_BGR2GRAY)

        # Display the face region in color and grayscale
        cv.imshow('Face (Color)', resized_face)
        cv.imshow('Face (Grayscale)', gray_face)

        # Calculate total pixels in the detected face
        total_face_pixels = w * h
        print(f'Face size: {total_face_pixels} pixels')

        # Define a grayscale range for masking
        gray_mean = np.mean(gray_face)
        min_gray = int(0.5 * gray_mean)
        max_gray = int(1.5 * gray_mean)

        # Create a mask for pixels within the defined range
        gray_mask = cv.inRange(gray_face, min_gray, max_gray)
        pixels_in_range = cv.countNonZero(gray_mask)

        # Display the mask
        cv.imshow('Gray Mask', gray_mask)

        # Print the count of pixels within the range
        print(f'Pixels in gray range [{min_gray}, {max_gray}]: {pixels_in_range}')

    return frame

while True:
    # Read the current frame from the video capture
    ret, frame = video_capture.read()
    if not ret:
        print("Failed to capture frame")
        break

    # Process the frame
    processed_frame = process_frame(frame)

    # Display the frame with rectangles around detected faces
    cv.imshow('Video Feed', processed_frame)

    # Exit loop if 'Esc' is pressed
    if cv.waitKey(1) == 27:
        break

# Release resources
video_capture.release()
cv.destroyAllWindows()
