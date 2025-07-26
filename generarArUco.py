import cv2
import cv2.aruco as aruco
import numpy as np
import os

# Crear carpeta para marcadores si no existe
output_folder = "aruco_markers"
os.makedirs(output_folder, exist_ok=True)

# Definir el diccionario ArUco. Usamos el mismo que en la aplicación principal.
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)

# Tamaño en píxeles de la imagen del marcador (puedes ajustarlo)
marker_size = 200

# Generar marcadores con IDs 0, 1, 2
for i in range(3):
    # Usar aruco.generateMarker() en lugar de aruco.generateImage()
    marker_image = aruco.generateImageMarker(aruco_dict, i, marker_size)
    
    # Guardar la imagen del marcador
    cv2.imwrite(os.path.join(output_folder, f"marker_id_{i}.png"), marker_image)
    print(f"Marcador ID {i} guardado en {output_folder}/marker_id_{i}.png")

print("\nScript de generación de marcadores ArUco completado.")
print("Por favor, imprime estos marcadores para usarlos con la aplicación de RA.")
