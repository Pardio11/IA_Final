import os
import cv2
from PIL import Image, ImageOps
import math

def procesar_detecciones_completas(img, nombre_base, ruta_salida, limite_por_imagen):
    """
    Aplica las transformaciones de rotación e inversión horizontal a toda la imagen original.
    """
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    img_reducida = img_pil.resize((60, 60))  # Reducir la imagen completa a 40x40

    # Parámetros de transformación
    angulos = [0, 90, 180, 270]
    transformaciones = []

    # Generar combinaciones de rotaciones e inversiones
    for angulo in angulos:
        img_rotada = img_reducida.rotate(angulo, expand=True)
        transformaciones.append(("rot", img_rotada))
        
        # Invertir horizontalmente
        img_invertida = ImageOps.mirror(img_rotada)
        transformaciones.append(("inv", img_invertida))

    # Guardar imágenes generadas
    contador_variaciones = 0
    for trans_id, (trans_nombre, img_trans) in enumerate(transformaciones):
        if contador_variaciones >= limite_por_imagen:
            break

        # Guardar cada transformación
        nombre_trans = f"{nombre_base}{trans_nombre}{trans_id}.png"
        img_trans.save(os.path.join(ruta_salida, nombre_trans))
        contador_variaciones += 1

def procesar_imagenes():
    # Solicitar rutas por consola
    ruta_entrada = input("Introduce la ruta de la carpeta de entrada: ").strip()
    ruta_salida = input("Introduce la ruta de la carpeta de salida: ").strip()

    # Archivo Haar Cascade predefinido (en la misma carpeta del script)
    haarcascade_path = os.path.join(os.path.dirname(__file__), "haarcascade_car.xml")

    # Verificar rutas
    if not os.path.exists(ruta_entrada):
        print(f"La ruta de entrada {ruta_entrada} no existe.")
        return
    if not os.path.exists(haarcascade_path):
        print(f"El archivo Haar Cascade {haarcascade_path} no existe.")
        return

    # Crear la carpeta de salida si no existe
    os.makedirs(ruta_salida, exist_ok=True)

    # Cargar el Haar Cascade
    car_cascade = cv2.CascadeClassifier(haarcascade_path)

    # Listar imágenes de entrada
    imagenes = [f for f in os.listdir(ruta_entrada) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
    total_imagenes = len(imagenes)
    if total_imagenes == 0:
        print("No se encontraron imágenes en la carpeta de entrada.")
        return

    # Calcular límite de variaciones por imagen
    limite_total = 12000
    limite_por_imagen = math.ceil(limite_total / total_imagenes)
    print(f"Límite de variaciones por imagen: {limite_por_imagen}")

    # Procesar cada imagen
    for archivo in imagenes:
        ruta_archivo = os.path.join(ruta_entrada, archivo)
        try:
            # Cargar la imagen
            img = cv2.imread(ruta_archivo)
            img_gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Detectar autos
            detecciones = car_cascade.detectMultiScale(img_gris, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            if len(detecciones) > 0:
                print(f"Procesando {archivo} - {len(detecciones)} autos detectados")
                nombre_base = os.path.splitext(archivo)[0]
                procesar_detecciones_completas(img, nombre_base, ruta_salida, limite_por_imagen)
            else:
                print(f"No se detectaron autos en {archivo}")
        except Exception as e:
            print(f"Error al procesar {archivo}: {e}")

# Ejecutar el script
if __name__ == "__main__":
    procesar_imagenes()