import cv2
import os
import re
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def process_and_detect_main_car(source_dir, output_dir, yolo_weights, yolo_cfg, coco_names, image_size=(80, 80)):
    # Crear la carpeta de salida si no existe
    os.makedirs(output_dir, exist_ok=True)

    # Obtener el índice inicial basado en los archivos existentes
    existing_files = [f for f in os.listdir(output_dir) if f.startswith("Mustang") and f.endswith(".jpg")]
    if existing_files:
        last_index = max([int(re.search(r"Mustang_(\d+)", f).group(1)) for f in existing_files])
    else:
        last_index = -1  # No hay archivos existentes

    frame_count = last_index + 1

    # Configuración de YOLO
    net = cv2.dnn.readNet(yolo_weights, yolo_cfg)
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    # Cargar las clases desde COCO
    with open(coco_names, "r") as f:
        classes = f.read().strip().split("\n")

    # Configuración del generador de imágenes
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # Procesar cada imagen en la carpeta de origen
    for file_name in os.listdir(source_dir):
        image_path = os.path.join(source_dir, file_name)
        image = cv2.imread(image_path)
        if image is None:
            continue

        print(f"Processing image: {file_name}")
        height, width, _ = image.shape
        blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        detections = net.forward(output_layers)

        # Lista para almacenar detecciones de "car"
        car_detections = []

        for detection in detections:
            for obj in detection:
                scores = obj[5:]
                class_id = int(np.argmax(scores))
                confidence = scores[class_id]

                if classes[class_id] == "car" and confidence > 0.5:
                    # Obtener coordenadas del bounding box
                    center_x, center_y, w, h = (obj[0:4] * [width, height, width, height]).astype("int")
                    x = max(0, int(center_x - w / 2))
                    y = max(0, int(center_y - h / 2))
                    w = min(w, width - x)
                    h = min(h, height - y)
                    area = w * h

                    # Guardar detección válida
                    car_detections.append((x, y, w, h, area))

        # Si hay detecciones, elegir la de mayor área (carro principal)
        if car_detections:
            car_detections.sort(key=lambda d: d[4], reverse=True)  # Ordenar por área descendente
            x, y, w, h, _ = car_detections[0]  # Tomar la detección principal

            # Recortar y procesar la imagen del carro principal
            cropped_car = image[y:y+h, x:x+w]
            resized_car = cv2.resize(cropped_car, image_size)

            # Aumento de datos con ImageDataGenerator
            datagen.fit(resized_car.reshape((1, *image_size, 3)))  # Ajustar el tamaño de la imagen para ImageDataGenerator

            # Guardar imágenes aumentadas
            images_saved = 0
            for _ in datagen.flow(resized_car.reshape((1, *image_size, 3)), batch_size=1, save_to_dir=output_dir, save_prefix="Mustang", save_format="jpg"):
                if images_saved >= 8:  # Cambiar a 8 para generar 8 imágenes
                    break
                images_saved += 1

    print(f"Processed and saved images to {output_dir}")

# Ejemplo de uso
source_dir = r"BackupOriginals\Mustang"  # Carpeta de imágenes de origen
output_dir = r"src\play\Mustang"  # Carpeta de salida

yolo_weights = "src/yolov3.weights"  # Archivo de pesos de YOLO
yolo_cfg = "src/yolov3.cfg"  # Archivo de configuración de YOLO
coco_names = "src/coco.names"  # Archivo de clases COCO

process_and_detect_main_car(
    source_dir=source_dir,
    output_dir=output_dir,
    yolo_weights=yolo_weights,
    yolo_cfg=yolo_cfg,
    coco_names=coco_names,
    image_size=(80, 80)
)