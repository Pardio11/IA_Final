import cv2 as cv
import numpy as np

# Initialize webcam capture
capture = cv.VideoCapture(0)

while True:
    ret, frame = capture.read()
    if ret:
        cv.imshow('Original Frame', frame)

        # Get the dimensions of the frame
        height, width = frame.shape[:2]

        # Create a blank image for separating color channels
        blank_image = np.zeros((height, width), dtype='uint8')

        # Split the image into color channels
        blue, green, red = cv.split(frame)

        # Merge individual channels with blank images to isolate them
        blue_channel = cv.merge([blue, blank_image, blank_image])
        green_channel = cv.merge([blank_image, green, blank_image])
        red_channel = cv.merge([blank_image, blank_image, red])

        # Display the isolated color channels
        cv.imshow('Blue Channel', blue_channel)
        cv.imshow('Green Channel', green_channel)
        cv.imshow('Red Channel', red_channel)

        # Wait for key press and break if ESC is pressed
        key = cv.waitKey(1) & 0xFF
        if key == 27:
            break
    else:
        break

# Release the capture object and close all OpenCV windows
capture.release()
cv.destroyAllWindows()
