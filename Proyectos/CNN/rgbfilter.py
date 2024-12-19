import os
import cv2
import numpy as np

# Función para modificar los canales RGB
def modify_rgb_channels(image, r_shift, g_shift, b_shift):
    """ Modificar los canales RGB de la imagen """
    b, g, r = cv2.split(image)  # Separar los canales de color
    r = cv2.add(r, r_shift)  # Aplicar desplazamiento al canal rojo
    g = cv2.add(g, g_shift)  # Aplicar desplazamiento al canal verde
    b = cv2.add(b, b_shift)  # Aplicar desplazamiento al canal azul
    return cv2.merge([b, g, r])  # Volver a combinar los canales

# Función para generar variantes con diferentes tonos RGB
def generate_rgb_variants(source_dir):
    img_names = os.listdir(source_dir)

    # Lista de desplazamientos RGB para generar diferentes tonos
    rgb_shifts = [
        (10, 10, 0),  # Más amarillo (rojo + verde)
        (0, 10, 10),  # Más cian (verde + azul)
        (10, 0, 10)   # Más magenta (rojo + azul)
    ]

    for img_name in img_names:
        image_path = os.path.join(source_dir, img_name)

        # Verificar si es un archivo de imagen
        if not os.path.isfile(image_path):
            continue

        # Cargar la imagen
        image = cv2.imread(image_path)

        for idx, (r_shift, g_shift, b_shift) in enumerate(rgb_shifts):
            # Generar la variante con desplazamiento RGB
            modified_image = modify_rgb_channels(image, r_shift, g_shift, b_shift)

            # Crear el nombre del archivo de salida con un sufijo indicando la variante
            augmented_image_name = f"{os.path.splitext(img_name)[0]}_rgb_variant_{idx + 1}.jpg"
            output_path = os.path.join(source_dir, augmented_image_name)

            # Guardar la imagen modificada en la misma carpeta con calidad ajustada
            cv2.imwrite(output_path, modified_image, [cv2.IMWRITE_JPEG_QUALITY, 90])
            print(f"Imagen variante guardada: {output_path}")

if __name__ == "__main__":
    # Solicitar el directorio al usuario
    source_dir = input("Ingrese la ruta del directorio de imágenes: ").strip()
    if not os.path.isdir(source_dir):
        print("La ruta proporcionada no es un directorio válido.")
    else:
        generate_rgb_variants(source_dir)
