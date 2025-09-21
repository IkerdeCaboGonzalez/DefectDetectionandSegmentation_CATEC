import os
import numpy as np

from PIL import Image, UnidentifiedImageError

# Definimos una función para recortar las imágenes y eliminar los bordes.

def recortar_bordes(imagen, margen=0):
    ancho, alto = imagen.size
    x1 = int(ancho*margen)
    y1 = int(alto*margen)
    x2 = int(ancho*(1-margen))
    y2 = int(alto*(1-margen))
    return imagen.crop((x1, y1, x2, y2))

# Función paara procesar las imágenes, utilizando la función anterior de recorte


def procesar_imagenes(ruta, image_size):
    contenedor = []
    extensiones_validas = ('.jpg', '.jpeg', '.png', '.tif', '.tiff')
    for archivo in os.listdir(ruta):
        if not archivo.lower().endswith(extensiones_validas):
            continue  
        nombre_archivo = os.path.join(ruta, archivo)
        try:
            imagen = Image.open(nombre_archivo)
            if imagen.mode != 'RGB':
                imagen = imagen.convert('RGB')
            imagen = recortar_bordes(imagen)  # Recorta los bordes de la imagen
            imagen = imagen.resize(image_size)
            
            contenedor.append(imagen)
        except (UnidentifiedImageError, OSError):
            print(f"Archivo no válido o corrupto: {nombre_archivo}")
            continue

    return np.array(contenedor)
