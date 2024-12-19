import cv2 as cv

# Load the face detection classifier
face_cascade = cv.CascadeClassifier('haarcascade_frontalface_alt.xml')

# Initialize video capture
video_stream = cv.VideoCapture(0)

while True:
    # Capture frame-by-frame
    frame_received, frame = video_stream.read()

    if not frame_received:
        break

    # Convert the frame to grayscale for face detection
    gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    # Loop through each detected face
    for (x, y, w, h) in faces:
        # Draw a red rectangle around the face
        frame = cv.rectangle(frame, (x, y), (x + w, y + h), (234, 23, 23), 2)

        # Draw a green rectangle in the lower half of the face
        frame = cv.rectangle(frame, (x, int(y + h / 2)), (x + w, y + h), (0, 255, 0), 5)

        # Draw two black circles for the eyes
        left_eye_center = (x + int(w * 0.3), y + int(h * 0.4))
        right_eye_center = (x + int(w * 0.7), y + int(h * 0.4))
        frame = cv.circle(frame, left_eye_center, 21, (0, 0, 0), 2)
        frame = cv.circle(frame, right_eye_center, 21, (0, 0, 0), 2)

        # Draw white circles for the eye pupils
        frame = cv.circle(frame, left_eye_center, 20, (255, 255, 255), -1)
        frame = cv.circle(frame, right_eye_center, 20, (255, 255, 255), -1)

        # Draw red dots for the eye pupils
        frame = cv.circle(frame, left_eye_center, 5, (0, 0, 255), -1)
        frame = cv.circle(frame, right_eye_center, 5, (0, 0, 255), -1)

    # Display the frame with face detection and eye markers
    cv.imshow('Face Detection', frame)

    # Break the loop if 'q' is pressed
    if cv.waitKey(1) == ord('q'):
        break

# Release the video capture and close any open windows
video_stream.release()
cv.destroyAllWindows()
