import cv2 as cv

import numpy as np
import time

img = cv.imread('salida.jpg', 1)
img2 = cv.cvtColor(img, cv.COLOR_BGR2RGB)
img3 = cv.cvtColor(img2, cv.COLOR_RGB2HSV)

img4 = cv.cvtColor(img, cv.COLOR_BGR2HSV)

#mascara roja
umbralBajo=(0, 80, 80)
umbralAlto=(10, 255, 255)
mascara1 = cv.inRange(img, umbralBajo, umbralAlto)
umbralBajoB=(170, 80,80)
umbralAltoB=(180, 255, 255)
mascara2 = cv.inRange(img, umbralBajoB, umbralAltoB)

mascaraRojo = mascara1 + mascara2

#mascara Verde
umbralBajo=(0, 0, 80)
umbralAlto=(0, 0, 255)
mascara1 = cv.inRange(img2, umbralBajo, umbralAlto) 
umbralBajoB=(35, 80,80)
umbralAltoB=(85, 255, 255)
mascara2 = cv.inRange(img2, umbralBajoB, umbralAltoB)

mascaraVerde = mascara1 + mascara2


resultado = cv.bitwise_and(img, img, mask=mascaraRojo)

cv.imshow('resultadoRojo', resultado)
cv.imshow('mascaraRojo', mascaraRojo)
cv.imshow('mascaraVerde', mascaraVerde)
cv.imshow('Img Original',img)
cv.imshow('Img BGR->RGB (ROJO)', img2)
cv.imshow('Img BGR->RGB->HSV (ROJO)', img3)
cv.imshow('Img BGR->HSV', img4)
cv.waitKey(0)
cv.destroyAllWindows()