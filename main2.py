import cv2
import cv2.aruco as aruco
import numpy as np
import os
import time

# --- 1. Parámetros de Calibración de Cámara ---
# Estos son valores de ejemplo. Para una aplicación real, debes calibrar tu cámara
# para obtener valores precisos.
# Puedes usar un script de calibración de cámara (ej. con un tablero de ajedrez).
# camera_matrix = np.array([[fx, 0, cx],
#                           [0, fy, cy],
#                           [0, 0, 1]], dtype=np.float32)
# dist_coeffs = np.array([k1, k2, p1, p2, k3], dtype=np.float32)
# Ejemplo de parámetros de una cámara genérica (ajusta según tu cámara)
camera_matrix = np.array([[800, 0, 320],
                          [0, 800, 240],
                          [0, 0, 1]], dtype=np.float32)
dist_coeffs = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32) # Sin distorsión para simplificar

# --- 2. Carga de Activos Digitales ---

# 2.1. Imagen 2D para el Marcador 0
# Asegúrate de que la ruta sea correcta.
image_2d_path = 'assets/logo.png' # Reemplaza con la ruta a tu imagen 2D
if not os.path.exists(image_2d_path):
    print(f"Advertencia: La imagen 2D no se encontró en '{image_2d_path}'. Usando una imagen de marcador de posición.")
    # Crear una imagen de marcador de posición si el archivo no existe
    image_2d = np.zeros((200, 200, 3), dtype=np.uint8)
    cv2.putText(image_2d, "2D Image", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
else:
    image_2d = cv2.imread(image_2d_path)
    if image_2d is None:
        print(f"Error: No se pudo cargar la imagen 2D desde '{image_2d_path}'. Usando una imagen de marcador de posición.")
        image_2d = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.putText(image_2d, "2D Image Error", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)


# 2.2. Sprite 2D Animado para el Marcador 1
# Carga las imágenes del sprite. Asume que están numeradas secuencialmente.
sprite_folder = 'assets/sprite_frames/' # Reemplaza con la ruta a tu carpeta de frames del sprite
sprite_frames = []
if os.path.exists(sprite_folder) and os.path.isdir(sprite_folder):
    frame_files = sorted([f for f in os.listdir(sprite_folder) if f.endswith(('.png', '.jpg'))])
    for frame_file in frame_files:
        frame_path = os.path.join(sprite_folder, frame_file)
        frame = cv2.imread(frame_path)
        if frame is not None:
            sprite_frames.append(frame)
        else:
            print(f"Advertencia: No se pudo cargar el frame del sprite '{frame_path}'.")
else:
    print(f"Advertencia: La carpeta de sprites no se encontró en '{sprite_folder}'. El sprite animado no estará disponible.")

if not sprite_frames:
    # Crear frames de marcador de posición si no hay sprites cargados
    for i in range(5):
        frame_placeholder = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.putText(frame_placeholder, f"Sprite Frame {i}", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        sprite_frames.append(frame_placeholder)


sprite_frame_idx = 0
last_sprite_update_time = time.time()
sprite_animation_speed = 0.1 # Segundos por frame

# --- 2.3. Objeto 3D (Nudo de Toroide) para el Marcador 2 ---

def generate_torus_knot_points(R, r, p, q, num_segments=200):
    """
    Genera los puntos 3D para un nudo de toroide.
    R: Radio mayor del toro.
    r: Radio menor del toro.
    p, q: Enteros coprimos que definen el tipo de nudo.
    num_segments: Número de segmentos para aproximar la curva.
    """
    points = []
    for i in range(num_segments):
        theta = 2 * np.pi * i / num_segments
        # Ecuaciones paramétricas para un nudo de toroide
        x = (R + r * np.cos(q * theta)) * np.cos(p * theta)
        y = (R + r * np.cos(q * theta)) * np.sin(p * theta)
        z = r * np.sin(q * theta) # Ajustado para que aparezca sobre el marcador
        points.append([x, y, z])
    return np.array(points, dtype=np.float32)

# Parámetros para el nudo de toroide
torus_R = 0.03 # Radio mayor (en metros, relativo al tamaño del marcador)
torus_r = 0.01 # Radio menor (en metros)
torus_p = 3    # Primer entero coprimo (ej. 3 para un nudo 3,2)
torus_q = 2    # Segundo entero coprimo (ej. 2 para un nudo 3,2)
num_torus_segments = 200 # Número de puntos para definir el nudo

# Generar los puntos del nudo de toroide
torus_knot_points = generate_torus_knot_points(torus_R, torus_r, torus_p, torus_q, num_torus_segments)


# --- 3. Funciones de Superposición ---

def overlay_image_2d(frame, image_to_overlay, marker_corners, marker_id):
    """
    Superpone una imagen 2D sobre el marcador detectado usando una transformación homográfica.
    """
    h, w, _ = image_to_overlay.shape
    # Puntos de la imagen a superponer (esquinas)
    img_pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    # Puntos del marcador en el frame (esquinas detectadas)
    dst_pts = marker_corners[0].astype(np.float32)

    # Calcular la matriz de homografía
    M, _ = cv2.findHomography(img_pts, dst_pts)

    # Aplicar la homografía a la imagen a superponer
    warped_image = cv2.warpPerspective(image_to_overlay, M, (frame.shape[1], frame.shape[0]))

    # Crear una máscara para la imagen superpuesta
    mask = np.zeros(frame.shape, dtype=np.uint8)
    cv2.fillConvexPoly(mask, dst_pts.astype(int), (255, 255, 255))
    mask_inv = cv2.bitwise_not(mask)

    # Combinar el frame original con la imagen superpuesta usando la máscara
    frame_bg = cv2.bitwise_and(frame, mask_inv)
    frame_fg = cv2.bitwise_and(warped_image, mask)
    combined_frame = cv2.add(frame_bg, frame_fg)

    return combined_frame

def draw_3d_object(frame, rvec, tvec, camera_matrix, dist_coeffs, object_points):
    """
    Dibuja un objeto 3D (nudo de toroide) proyectando sus puntos 3D sobre el frame 2D.
    """
    # Proyectar los puntos 3D del objeto al plano 2D de la imagen
    imgpts, jac = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, dist_coeffs)
    imgpts = np.int32(imgpts).reshape(-1, 2)

    # Dibujar líneas conectando puntos consecutivos para formar el nudo
    for i in range(len(imgpts) - 1):
        cv2.line(frame, tuple(imgpts[i]), tuple(imgpts[i+1]), (255, 0, 255), 2) # Color magenta
    # Cerrar el bucle para conectar el último punto con el primero
    cv2.line(frame, tuple(imgpts[-1]), tuple(imgpts[0]), (255, 0, 255), 2)
    
    return frame

# --- 4. Configuración de ArUco ---
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
parameters = aruco.DetectorParameters()
detector = aruco.ArucoDetector(aruco_dict, parameters)

# Define los puntos 3D de un marcador ArUco en su propio sistema de coordenadas.
# Asumimos que el origen está en el centro del marcador y que el marcador
# tiene un tamaño de 0.05 metros (5 cm) de lado.
marker_size_m = 0.05 # Este valor DEBE coincidir con el tamaño real de tu marcador impreso en metros.
half_marker_size = marker_size_m / 2.0
marker_obj_points = np.array([
    [-half_marker_size, half_marker_size, 0],   # Esquina superior izquierda
    [ half_marker_size, half_marker_size, 0],    # Esquina superior derecha
    [ half_marker_size, -half_marker_size, 0],   # Esquina inferior derecha
    [-half_marker_size, -half_marker_size, 0]    # Esquina inferior izquierda
], dtype=np.float32)


# --- 5. Bucle Principal de la Aplicación de RA ---
def run_ar_application():
    global sprite_frame_idx, last_sprite_update_time

    cap = cv2.VideoCapture(0) # 0 para la cámara predeterminada. Cambia si tienes múltiples cámaras.

    if not cap.isOpened():
        print("Error: No se pudo abrir la cámara.")
        return

    print("Aplicación de Realidad Aumentada iniciada. Presiona 'q' para salir.")
    print("Asegúrate de tener los marcadores ArUco (IDs 0, 1, 2) y los archivos de assets.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: No se pudo leer el frame de la cámara. Saliendo...")
            break

        # Convertir a escala de grises para la detección de ArUco
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detectar marcadores
        corners, ids, rejectedImgPoints = detector.detectMarkers(gray)

        # Si se detectan marcadores
        if ids is not None:
            for i in range(len(ids)):
                marker_id = ids[i][0]
                marker_corners_2d = corners[i][0] # Obtener los puntos 2D del marcador

                # Dibujar el contorno del marcador y su ID
                cv2.polylines(frame, [np.int32(marker_corners_2d)], True, (0, 255, 255), 2)
                cv2.putText(frame, f"ID: {marker_id}", tuple(np.int32(marker_corners_2d[0])),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

                # Estimar la pose del marcador usando solvePnP
                # marker_obj_points son los puntos 3D del marcador en su propio sistema de coordenadas
                # marker_corners_2d son los puntos 2D correspondientes en la imagen
                success, rvec, tvec = cv2.solvePnP(marker_obj_points, marker_corners_2d, camera_matrix, dist_coeffs)
                
                if success:
                    # Dibujar los ejes de coordenadas para visualizar la pose 3D
                    cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.03) # Longitud de los ejes

                    # Superponer objetos según el ID del marcador
                    if marker_id == 0:
                        # Superponer imagen 2D
                        if image_2d is not None:
                            frame = overlay_image_2d(frame, image_2d, corners[i], marker_id) # Usar corners[i] para overlay_image_2d
                        else:
                            cv2.putText(frame, "Imagen 2D no cargada", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    elif marker_id == 1:
                        # Superponer sprite 2D animado
                        if sprite_frames:
                            current_time = time.time()
                            if current_time - last_sprite_update_time > sprite_animation_speed:
                                sprite_frame_idx = (sprite_frame_idx + 1) % len(sprite_frames)
                                last_sprite_update_time = current_time

                            current_sprite_frame = sprite_frames[sprite_frame_idx]
                            frame = overlay_image_2d(frame, current_sprite_frame, corners[i], marker_id) # Usar corners[i] para overlay_image_2d
                        else:
                            cv2.putText(frame, "Sprite no cargado", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    elif marker_id == 2:
                        # Superponer objeto 3D (nudo de toroide)
                        frame = draw_3d_object(frame, rvec, tvec, camera_matrix, dist_coeffs, torus_knot_points)
                    else:
                        cv2.putText(frame, f"Marcador ID {marker_id} detectado", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                else:
                    cv2.putText(frame, f"No se pudo estimar pose para ID {marker_id}", tuple(np.int32(marker_corners_2d[0]) + np.array([0, 20])),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)


        # Mostrar el frame resultante
        cv2.imshow('Aplicacion de Realidad Aumentada', frame)

        # Salir si se presiona 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Liberar recursos
    cap.release()
    cv2.destroyAllWindows()

# Ejecutar la aplicación
if __name__ == '__main__':
    # Crear las carpetas de assets si no existen
    os.makedirs('assets', exist_ok=True)
    os.makedirs('assets/sprite_frames', exist_ok=True)
    print("Asegúrate de colocar 'logo.png' en la carpeta 'assets/' y los frames de tu sprite en 'assets/sprite_frames/'.")
    print("También puedes generar tus propios marcadores ArUco con IDs 0, 1 y 2.")
    run_ar_application()
