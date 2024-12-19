import numpy as np
import cv2 as cv
import os
import glob

# Check if the directory exists, if not, create it
if not os.path.exists('imgs'):
    os.makedirs('imgs')

rostro = cv.CascadeClassifier('haarcascade_frontalface_alt.xml')
cap = cv.VideoCapture(0)
x=y=w=h= 0 
img = 0
count = 0
while True:
    ret, frame = cap.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    rostros = rostro.detectMultiScale(gray, 1.3, 5)
    face=None
    for(x, y, w, h) in rostros:
        m= int(h/2)
        #Boca
        #frame = cv.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 2)
        #Cara
        #frame = cv.rectangle(frame, (x,y+m), (x+w, y+h), (255, 0 ,0), 2 )
        img = 180- frame[y:y+h,x:x+w]
        face=frame[y:y+h,x:x+w]
        count = count + 1   
    if face is None:
        continue
    # Check if the image is suitable for resizing
    if face.shape[0] >= 100 and face.shape[1] >= 100:
        # Resize the image to 100x100 and save it in grayscale
        face_resized_100 = cv.resize(face, (100, 100))
        face_resized_100_gray = cv.cvtColor(face_resized_100, cv.COLOR_BGR2GRAY)
        name = 'imgs/face/100/cara'+str(count)+'.jpg'
        cv.imwrite(name, face_resized_100_gray)

    if face.shape[0] >= 80 and face.shape[1] >= 80:
        # Resize the image to 80x80 and save it in grayscale
        face_resized_80 = cv.resize(face, (80, 80))
        face_resized_80_gray = cv.cvtColor(face_resized_80, cv.COLOR_BGR2GRAY)
        name = 'imgs/face/80/cara'+str(count)+'.jpg'
        cv.imwrite(name, face_resized_80_gray)

    cv.imshow('rostros', frame)
    cv.imshow('cara', img)
    
    k = cv.waitKey(1)
    if k == 27:
        break
cap.release()
cv.destroyAllWindows()

# Get the list of all image files in the directory
image_files = glob.glob('imgs/face/*/*.jpg')

# Define the threshold value
threshold_value = 127

for image_file in image_files:
    # Read the image
    img = cv.imread(image_file, cv.IMREAD_GRAYSCALE)

    # Apply the threshold
    _, img_thresholded = cv.threshold(img, threshold_value, 255, cv.THRESH_BINARY)

    # Create a new path for the thresholded image
    new_path = image_file.replace('imgs/face', 'imgs/face/bw')

    # Create directories if they don't exist
    os.makedirs(os.path.dirname(new_path), exist_ok=True)

    # Save the thresholded image
    cv.imwrite(new_path, img_thresholded)