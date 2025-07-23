import cv2
import mediapipe as mp
import math
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
import os
import ffmpeg
from scipy.signal import find_peaks, find_peaks
from scipy.ndimage import uniform_filter1d
import logging
from PIL import Image, ImageDraw, ImageFont

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s",
                    handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)

def plot_3d_landmarks(results):
    """
    Función para graficar los puntos 3D predichos por MediaPipe con conexiones.
    """
    # Verificar si se detectaron landmarks
    if not results.pose_landmarks:
        print("No se detectaron landmarks para graficar.")
        return

    # Extraer las coordenadas x, y, z
    landmarks = results.pose_landmarks.landmark
    x_coords = [landmark.x for landmark in landmarks]
    y_coords = [landmark.y for landmark in landmarks]
    z_coords = [landmark.z for landmark in landmarks]

    # Crear el gráfico 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Graficar los puntos en 3D
    ax.scatter(x_coords, y_coords, z_coords, c='r', marker='o')  # Color rojo, marcador círculo

    # Dibujar las conexiones
    mp_pose = mp.solutions.pose
    for connection in mp_pose.POSE_CONNECTIONS:
        start_idx, end_idx = connection
        start_point = landmarks[start_idx]
        end_point = landmarks[end_idx]

        # Coordenadas de inicio y fin
        x = [start_point.x, end_point.x]
        y = [start_point.y, end_point.y]
        z = [start_point.z, end_point.z]

        # Graficar línea entre los puntos
        ax.plot(x, y, z, c='b', linewidth=1)  # Color azul, línea delgada

    # Etiquetas de los ejes
    ax.set_xlabel('Eje X')
    ax.set_ylabel('Eje Y')
    ax.set_zlabel('Eje Z')

    # Ajustar perspectiva (invertir ejes si es necesario)
    ax.invert_zaxis()
    ax.invert_yaxis()

    # Mostrar el gráfico
    plt.show()

def scale_landmarks(landmarks, w, h):
    """
    Escala las coordenadas normalizadas de los puntos clave a píxeles.

    Parámetros:
        landmarks (list): Lista de puntos clave normalizados.
        w (int): Ancho de la imagen.
        h (int): Altura de la imagen.

    Retorna:
        list: Lista de puntos clave escalados a píxeles.
    """
    scaled_landmarks = []
    for point in landmarks:
        scaled_point = type(point)(
            x=point.x * w,
            y=point.y * h,
            z=point.z,
            visibility=point.visibility
        )
        scaled_landmarks.append(scaled_point)
    return scaled_landmarks

def draw_lines_and_angle(image, points_group, line_color=(180, 180, 30), angle_color=(180, 180, 30),
                         text_color=(255, 255, 255), alpha=0.5, back_angle=False):
    """
    Dibuja las líneas AB y BC, el ángulo y una elipse representando el ángulo entre las líneas.

    Parámetros:
        image: Imagen en la que se dibujarán las líneas y el ángulo.
        points_group (list): Lista de 1 o mas grupos de tres puntos en 2D como [(x, y), (x, y), (x, y)].
    """
    max_z = 0
    id_max_z = 0
    for idx, group in enumerate(points_group):
        points = group
        # Extraer los puntos desde el diccionario
        pointA = (int(points[0].x), int(points[0].y))
        pointB = (int(points[1].x), int(points[1].y))
        pointC = (int(points[2].x), int(points[2].y))


        mean_z = (points[0].z + points[1].z + points[2].z) / 3
        if mean_z < max_z:
            max_z = mean_z
            id_max_z = idx

    points = points_group[id_max_z]
    # Extraer los puntos desde el diccionario
    pointA = (int(points[0].x), int(points[0].y))
    pointB = (int(points[1].x), int(points[1].y))
    pointC = (int(points[2].x), int(points[2].y))

    cv2.line(image, pointA, pointB, line_color, 2)
    cv2.line(image, pointB, pointC, line_color, 2)

    # Dibuja los puntos A, B y C
    points_color = tuple(int(c * 0.65) for c in line_color)
    cv2.circle(image, pointA, 5, points_color, -1)  
    cv2.circle(image, pointB, 5, points_color, -1)
    cv2.circle(image, pointC, 5, points_color, -1)

    angle = get_angle(points)

    #radius = int(shortest_segment_value(pointA, pointB, pointC) * 0.85)
    radius = 30
    axes = (radius, radius)  # Ejes de la elipse
    center = pointB  # Centro de la elipse en el punto B

    # Calcular los ángulos de inicio y fin
    start_angle = np.degrees(np.arctan2(pointA[1] - pointB[1], pointA[0] - pointB[0]))
    end_angle = np.degrees(np.arctan2(pointC[1] - pointB[1], pointC[0] - pointB[0]))

    # Se suman 360 grados si el ángulo trasero
    if back_angle:
        start_angle += 360

    # Determinar un ángulo intermedio (en el centro del barrido)
    middle_angle = np.radians((start_angle + end_angle) / 2)

    # Calcular la posición del texto a lo largo de la elipse
    text_x = int(center[0] + radius * 1.1 * np.cos(middle_angle)) # NOTA: Se debe ajustar si está en el lado izquierdo
    if back_angle:
        text_x -= cv2.getTextSize(f"{angle:.2f}", cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0][0]   
    text_y = int(center[1] + radius * 1.1 * np.sin(middle_angle))
    text_position = (text_x, text_y)

    # Crear una copia de la imagen original para la máscara
    overlay = image.copy()

    # Dibujar la elipse en la máscara con relleno sólido
    cv2.ellipse(overlay, center, axes, 0, start_angle, end_angle, angle_color, -1)  # -1 para relleno

    # Fusionar la imagen original con la máscara usando la transparencia `alpha`
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

    # Dibujar la elipse
    cv2.ellipse(image, center, axes, 0, start_angle, end_angle, angle_color, 2)

    # Dibujar un rectángulo detrás del texto
    text_size = cv2.getTextSize(f"{angle:.2f}", cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
    # Calcular nuevas coordenadas con margen
    margin = 2
    text_position_start = (text_position[0] - margin, text_position[1] + margin)
    text_position_end = (text_position[0] + text_size[0] + margin, text_position[1] - text_size[1] - margin)

    cv2.rectangle(image, text_position_start, text_position_end, angle_color, -1)

    # Dibujar el texto del ángulo en la trayectoria de la elipse
    cv2.putText(image, f"{angle:.2f}", text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
    
    return angle

def shortest_segment_value(A, B, C):
    """
    Retorna la distancia del segmento más corto entre AB y BC.

    Parámetros:
        A (tuple): Coordenadas del punto A (x, y).
        B (tuple): Coordenadas del punto B (x, y).
        C (tuple): Coordenadas del punto C (x, y).

    Retorna:
        float: La distancia del segmento más corto.
    """
    def euclidean_distance(P1, P2):
        return np.linalg.norm(np.array(P1) - np.array(P2))

    dist_AB = euclidean_distance(A, B)
    dist_BC = euclidean_distance(B, C)

    return min(dist_AB, dist_BC)

def get_angle(points):
    """
    Calcula el ángulo mínimo (en grados) entre dos vectores en 3D formados por tres puntos: 
    B-A y B-C.
    
    Parámetros:
        points (list): Lista de tres puntos en 3D como [Point(x, y, z), Point(x, y, z), Point(x, y, z)].
    
    Retorna:
        float: Ángulo mínimo en grados.
    """
    # Convertir las coordenadas de los puntos a vectores
    pointA = (points[0].x, points[0].y, points[0].z)
    pointB = (points[1].x, points[1].y, points[1].z)
    pointC = (points[2].x, points[2].y, points[2].z)
    
    # Vectores BA y BC
    vectorBA = (pointA[0] - pointB[0], pointA[1] - pointB[1], pointA[2] - pointB[2])
    vectorBC = (pointC[0] - pointB[0], pointC[1] - pointB[1], pointC[2] - pointB[2])
    
    # Producto escalar de AB y BC
    dot_product = (vectorBA[0] * vectorBC[0] + 
                   vectorBA[1] * vectorBC[1] + 
                   vectorBA[2] * vectorBC[2])
    
    # Magnitudes de los vectores AB y BC
    magnitude_BA = math.sqrt(vectorBA[0]**2 + vectorBA[1]**2 + vectorBA[2]**2)
    magnitude_BC = math.sqrt(vectorBC[0]**2 + vectorBC[1]**2 + vectorBC[2]**2)
    
    # Verificar si alguna magnitud es cero para evitar división por cero
    if magnitude_BA == 0 or magnitude_BC == 0:
        raise ValueError("Uno de los vectores tiene magnitud cero, no se puede calcular el ángulo.")
    
    # Coseno del ángulo
    cos_theta = dot_product / (magnitude_BA * magnitude_BC)
    
    # Asegurar que el valor de cos_theta esté en el rango [-1, 1] para evitar errores numéricos
    cos_theta = max(-1.0, min(1.0, cos_theta))
    
    # Ángulo en radianes
    angle_rad = math.acos(cos_theta)
    
    # Convertir el ángulo a grados
    angle_deg = math.degrees(angle_rad)
    
    # Retornar el ángulo mínimo
    return angle_deg

def detect_objects(source, conf=0.3, line_width=2, classes=None):
    """
    Detect objects in an image or frame using YOLO.

    Parameters:
        source: str or np.ndarray
            The source image file path or a frame (as a NumPy array).
        conf: float
            The confidence threshold for predictions.

    Returns:
        bounding_boxes: list
            A list of dictionaries containing bounding box information.
    """
    logger.info("Cargando el modelo YOLO...")
    model = YOLO('yolo_best.pt')
    logger.info("Modelo YOLO cargado exitosamente.")
    # Realiza la predicción
    results = model.predict(source=source, conf=conf, save=False, line_width=line_width, verbose=False)

    bounding_boxes = []  # Lista para almacenar las bounding boxes

    for result in results:
        for box in result.boxes:
            if classes is None:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())  # Coordenadas de la caja
                x, y, w, h = x1, y1, x2 - x1, y2 - y1
                confidence = box.conf[0].item()  # Confianza del modelo
                class_id = int(box.cls[0].item())  # Clase predicha
                bounding_boxes.append({
                    "x": x, "y": y, "w": w, "h": h,
                    "confidence": confidence, "class_id": class_id
                })
            elif int(box.cls[0].item()) == classes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                x, y, w, h = x1, y1, x2 - x1, y2 - y1
                confidence = box.conf[0].item()
                bounding_boxes.append({
                    "x": x, "y": y, "w": w, "h": h,
                    "confidence": confidence, "class_id": classes
                })
    return bounding_boxes

def analize_bbox(video_path, output_path, tracker):
    """
    Analiza el video para detectar un objeto y realizar seguimiento del mismo.

    Parámetros:
        video_path (str): Ruta del video a analizar.
        output_path (str): Ruta de la carpeta donde se guardarán los resultados.
        tracker: Objeto del tracker de OpenCV.

    Retorna:
        frame_numbers (list): Lista de números de fotogramas donde se detectó el objeto.
        bbox_heights (list): Lista de alturas de las cajas delimitadoras en cada fotograma.
        initial_frame (int): Número del primer fotograma donde se detectó el objeto.
        bboxes (list): Lista de cajas delimitadoras detectadas en cada fotograma.
    """
    frame_numbers = []
    bbox_heights = []
    bboxes = []
    initial_frame = 0

    video_name = os.path.splitext(os.path.basename(video_path))[0]

    video_folder = os.path.join(output_path, video_name)
    os.makedirs(video_folder, exist_ok=True)

    video_avi = os.path.join(video_folder, "video_tracker.avi")
    video_mp4 = os.path.join(video_folder, "video_tracker.mp4")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Error: No se pudo abrir el video {video_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec para .avi

    if width > height:
        logger.info("El video está en formato horizontal, se rotará 90° en sentido horario.")
        width, height = height, width
        rotate = True
    else:
        rotate = False
    
    out = cv2.VideoWriter(video_avi, fourcc, fps, (width, height))

    ret, frame = cap.read()
    if rotate:
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    bounding_boxes = detect_objects(source=frame, conf=0.15, line_width=2, classes=0)
    while len(bounding_boxes) == 0:
        ret, frame = cap.read()
        initial_frame += 1
        bounding_boxes = detect_objects(source=frame, conf=0.15, line_width=2, classes=0)
        if not ret:
            logger.warning("No se detectaron objetos en el video.")
            exit()
    bbox = (bounding_boxes[0]["x"], bounding_boxes[0]["y"], bounding_boxes[0]["w"], bounding_boxes[0]["h"])
    tracker.init(frame, bbox)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if rotate:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        bboxes.append(bbox)
        success, bbox = tracker.update(frame)

        if success:
            # Dibuja la caja delimitadora
            x, y, w, h = [int(i) for i in bbox]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2, 1)
            frame_numbers.append(len(frame_numbers) + 1)
            bbox_heights.append(y + h)
        else:
            cv2.putText(frame, "Tracking perdido", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

        out.write(frame)

    # Liberar recursos
    cap.release()
    out.release()
    
    ffmpeg.input(video_avi).output(video_mp4, vcodec="libx264", acodec="aac").overwrite_output().run()
    # Borra el archivo .avi 
    os.remove(video_avi)   

    logger.info(f"Video procesado y guardado en {video_mp4}")
    return frame_numbers, bbox_heights, initial_frame, bboxes

def find_flat_points(signal, peaks, height=None, distance=None, threshold=0.1, min_distance_from_peak=2):
    """
    Encuentra los puntos donde la derivada de la señal es cercana a cero, excluyendo aquellos cercanos a los picos.

    Parámetros:
        signal (array): La señal de entrada (altura del bbox, por ejemplo).
        peaks (array): Índices de los picos (máximos) en la señal.
        height (float o tuple): Umbral de altura para los puntos planos.
        distance (int): Distancia mínima entre puntos planos.
        threshold (float): Umbral para considerar que la derivada es cercana a cero.
        min_distance_from_peak (int): Distancia mínima permitida entre un flat_point y un pico.

    Retorna:
        flat_points (array): Índices de los puntos donde la derivada es cercana a cero.
        properties (dict): Propiedades de los puntos detectados.
    """
    # Calcular la derivada de la señal
    derivative = np.gradient(signal)

    # Encontrar los puntos donde la derivada es cercana a cero (dentro del umbral)
    flat_indices = np.where(np.abs(derivative) < threshold)[0]

    # Si no hay puntos cercanos a cero, retornar vacío
    if len(flat_indices) == 0:
        return np.array([]), {}

    # Crear una señal auxiliar con -abs(derivative) para usar find_peaks
    aux_signal = -np.abs(derivative)

    # Encontrar los "picos" en la señal auxiliar (que corresponden a valles en la derivada)
    flat_points, properties = find_peaks(
        aux_signal,
        height=height,
        distance=distance,
    )

    # Filtrar los puntos para asegurarnos de que estén en flat_indices
    flat_points = np.intersect1d(flat_points, flat_indices)

    # Excluir flat_points que estén demasiado cerca de los picos
    if len(peaks) > 0:
        # Crear una máscara para excluir puntos cercanos a los picos
        mask = np.ones_like(flat_points, dtype=bool)
        for peak in peaks:
            # Excluir puntos dentro de min_distance_from_peak frames del pico
            mask &= (np.abs(flat_points - peak) > min_distance_from_peak)
        
        # Aplicar la máscara para filtrar flat_points
        flat_points = flat_points[mask]

    return flat_points, properties

def detect_squat_repetitions(bbox_heights, frame_numbers, output_path, size=20,
                             distance=5, threshold=0.01, min_distance_from_peak=40):
    # 1. Suavizar la señal
    frame_numbers = np.array(frame_numbers)
    bbox_heights = np.array(bbox_heights)
    bbox_heights_smooth = uniform_filter1d(bbox_heights, size=size)
    
    # 2. Encontrar picos y valles
    peaks, _ = find_peaks(bbox_heights_smooth, height=np.mean(bbox_heights_smooth), distance=20)
    valleys, _ = find_flat_points(
        bbox_heights_smooth,
        height=np.mean(-np.abs(np.gradient(bbox_heights_smooth))),
        distance=distance,
        threshold=threshold,
        peaks=peaks,
        min_distance_from_peak=min_distance_from_peak
    )
    
    # 3. Identificar los puntos de inicio y fin de cada repetición
    start_frames = []
    end_frames = []
    
    for peak in peaks:
        start_frame = valleys[valleys < peak][-1] if any(valleys < peak) else None
        end_frame = valleys[valleys > peak][0] if any(valleys > peak) else None
        
        if start_frame is not None and end_frame is not None:
            start_frames.append(start_frame)
            end_frames.append(end_frame)
    # 4. Invirtiendo los valores en el eje y
    bbox_heights_smooth = -bbox_heights_smooth
    bbox_heights = -bbox_heights

    # 5. Graficar los resultados
    plt.figure(figsize=(12, 6))
    plt.plot(frame_numbers, bbox_heights_smooth, label="Altura del bbox (suavizada)", color='blue')
    #plt.plot(frame_numbers[peaks], bbox_heights_smooth[peaks], "v", label="Picos", color='red')
    #plt.plot(frame_numbers[valleys], bbox_heights_smooth[valleys], "^", label="Valles", color='green')
    
    for start, end in zip(start_frames, end_frames):
        plt.axvline(x=frame_numbers[start], color='orange', linestyle='--', label="Inicio repetición" if start == start_frames[0] else "")
        plt.axvline(x=frame_numbers[end], color='purple', linestyle='--', label="Fin repetición" if end == end_frames[0] else "")
    
    plt.title("Detección de repeticiones de sentadillas")
    plt.xlabel("Número de frame")
    plt.ylabel("Altura del bbox (y + h)")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{output_path}/rep_squat_graph.png")
    
    # # 6. Mostrar los fotogramas de inicio y fin de cada repetición
    # for i, (start, end) in enumerate(zip(start_frames, end_frames)):
    #     print(f"Repetición {i + 1}:")
    #     print(f"  - Inicio: Frame {frame_numbers[start]}")
    #     print(f"  - Pico: Frame {peaks[i]}")
    #     print(f"  - Fin:    Frame {frame_numbers[end]}")
    logger.info(f"Se detectaron {len(start_frames)} repeticiones de sentadillas.")
    logger.info(f"Gráfico de repeticiones guardado en {output_path}/rep_squat_graph.png")
    return start_frames, end_frames, peaks

def detect_deadlift_repetitions(bbox_heights, frame_numbers, output_path, size=20,
                                    distance=5, threshold=0.01, min_distance_from_peak=40):
        """
        Detecta repeticiones de peso muerto en un video basado en la altura del bbox.
        
        Parámetros:
            bbox_heights (list): Lista de alturas del bbox en cada frame.
            frame_numbers (list): Lista de números de frames correspondientes.
            output_path (str): Ruta donde se guardarán los resultados.
            size (int): Tamaño del filtro para suavizar la señal.
            distance (int): Distancia mínima entre picos detectados.
            threshold (float): Umbral para considerar un pico.
            min_distance_from_peak (int): Distancia mínima desde un pico para considerar un valle.
    
        Retorna:
            start_frames (list): Lista de frames donde comienzan las repeticiones.
            end_frames (list): Lista de frames donde terminan las repeticiones.
            peaks (list): Lista de picos detectados en la altura del bbox.
        """
        # 1. Suavizar la señal
        frame_numbers = np.array(frame_numbers)
        bbox_heights = np.array(bbox_heights)
        bbox_heights_smooth = uniform_filter1d(bbox_heights, size=size)
        
        # 2. Encontrar picos y valles
        peaks, _ = find_peaks(bbox_heights_smooth, height=np.mean(bbox_heights_smooth), distance=20)
        valleys, _ = find_flat_points(
            bbox_heights_smooth,
            height=np.mean(-np.abs(np.gradient(bbox_heights_smooth))),
            distance=distance,
            threshold=threshold,
            peaks=peaks,
            min_distance_from_peak=min_distance_from_peak
        )
        
        # 3. Identificar los puntos de inicio y fin de cada repetición
        start_frames = []
        end_frames = []
        
        for peak in peaks:
            start_frame = valleys[valleys < peak][-1] if any(valleys < peak) else None
            end_frame = valleys[valleys > peak][0] if any(valleys > peak) else None
            
            if start_frame is not None and end_frame is not None:
                start_frames.append(start_frame)
                end_frames.append(end_frame)
        
        # 4. Graficar los resultados
        plt.figure(figsize=(12, 6))
        plt.plot(frame_numbers, bbox_heights_smooth, label="Altura del bbox (suavizada)", color='blue')
        #plt.plot(frame_numbers[peaks], bbox_heights_smooth[peaks], "v", label="Picos", color='red')
        #plt.plot(frame_numbers[valleys], bbox_heights_smooth[valleys], "^", label="Valles", color='green')
        for start, end in zip(start_frames, end_frames):
            plt.axvline(x=frame_numbers[start], color='orange', linestyle='--', label="Inicio repetición" if start == start_frames[0] else "")
            plt.axvline(x=frame_numbers[end], color='purple', linestyle='--', label="Fin repetición" if end == end_frames[0] else "")
        plt.title("Detección de repeticiones de peso muerto")
        plt.xlabel("Número de frame")
        plt.ylabel("Altura del bbox (y + h)")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{output_path}/rep_deadlift_graph.png")
        logger.info(f"Gráfico de repeticiones guardado en {output_path}/rep_deadlift_graph.png")
        logger.info(f"Se detectaron {len(start_frames)} repeticiones de peso muerto.")
        return start_frames, end_frames, peaks

def create_video(video_path, output_path, exercise, start_frames,
                 end_frames, peaks, bboxes, sec_pause=3, initial_frame=0):
    '''
    Crea un video con las sentadillas detectadas, mostrando los ángulos de las rodillas y caderas.
    Parámetros:
        video_path (str): Ruta del video original.
        output_path (str): Ruta de la carpeta donde se guardará el video procesado.
        exercise (str): Nombre del ejercicio (debe ser "squat").
        start_frames (list): Lista de números de fotogramas donde comienza cada sentadilla.
        end_frames (list): Lista de números de fotogramas donde termina cada sentadilla.
        peaks (list): Lista de picos detectados en la altura del bbox.
        bboxes (list): Lista de cajas delimitadoras detectadas en cada fotograma.
        sec_pause (int): Tiempo en segundos para pausar entre sentadillas.
        initial_frame (int): Número del primer fotograma donde se detectó el objeto.
    Retorna:
        None'''

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    video_folder = os.path.join(output_path, video_name)
    os.makedirs(video_folder, exist_ok=True)

    video_avi = os.path.join(video_folder, "result.avi")
    video_mp4 = os.path.join(video_folder, "result.mp4")

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Error: No se pudo abrir el video {video_path}")  
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    
    if width > height:
        logger.info("El video está en formato horizontal, se rotará 90° en sentido horario.")
        width, height = height, width
        rotate = True
    else:
        rotate = False

    out = cv2.VideoWriter(video_avi, fourcc, fps, (width, height))

    # Fuente para el texto "Sentadilla N"
    font_path = "fonts/Roboto-Regular.ttf"
    font = ImageFont.truetype(font_path, size=32)

    # Saltar hasta el frame inicial
    for _ in range(initial_frame + 1):
        ret, frame = cap.read()

    frame_count = 0
    text_blocks = []
    sentadilla_index = 1

    # Almacena temporalmente los datos hasta que se complete el ejercicio
    current_exercise_data = {
        "angulo_rodilla": None,
        "angulo_cadera": None,
        "categoria": None,
        "feedback": None,
    }
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if rotate:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        if frame_count < len(bboxes):
            x, y, w, h = bboxes[frame_count]
            x, y, w, h = int(x), int(y), int(w), int(h)
            overlay = frame.copy()
            alpha = 0.3  # Nivel de transparencia (0.0 completamente transparente, 1.0 completamente opaco)

            # Dibujar el rectángulo en la superposición
            cv2.rectangle(overlay, (x, y), (x + w, y + h), (19, 245, 115), -1)

            # Combinar la superposición con el frame original
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

            # Dibujar el borde del rectángulo (opaco)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 245, 115), 2)

        angle_knee = angle_hip = 0

        if results.pose_landmarks:
            right_knee_points = scale_landmarks(
                [results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP.value],
                 results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE.value],
                 results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE.value]],
                width, height
            )
            left_knee_points = scale_landmarks(
                [results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP.value],
                 results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE.value],
                 results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE.value]],
                width, height
            )
            right_hip_points = scale_landmarks(
                [results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                 results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP.value],
                 results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE.value]],
                width, height
            )
            left_hip_points = scale_landmarks(
                [results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                 results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP.value],
                 results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE.value]],
                width, height
            )

            if frame_count == 0:
                A = (right_knee_points[1].x, right_knee_points[1].y)
                B = (right_knee_points[2].x, right_knee_points[2].y)
                C = (results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].x * width,
                     results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].y * height)
                determinante = (B[0] - A[0]) * (C[1] - A[1]) - (B[1] - A[1]) * (C[0] - A[0])
                back_angle_knee = determinante <= 0
                back_angle_hip = not back_angle_knee

            angle_knee = draw_lines_and_angle(frame, (right_knee_points, left_knee_points), back_angle=back_angle_knee)
            angle_hip = draw_lines_and_angle(frame, (right_hip_points, left_hip_points), back_angle=back_angle_hip)
        
        # Verificar si estamos en una sentadilla activa
        for i, (start, end) in enumerate(zip(start_frames, end_frames)):
            if start <= frame_count <= end:
                # Convertir a PIL para dibujar texto
                frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(frame_pil)
                text = f"Sentadilla {i+1}"

                text_bbox = draw.textbbox((0, 0), text, font=font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]

                x, y = 20, 20
                padding = 5
                rect_coords = [x - padding, y - padding, x + text_width + padding, y + text_height + (2 * padding)]

                # Dibujar el rectángulo negro detrás del texto
                draw.rectangle(rect_coords, fill=(0, 0, 0))

                # Dibujar el texto encima del rectángulo
                draw.text((x, y), text, font=font, fill=(255, 255, 255))

                # Convertir de nuevo a formato OpenCV
                frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)
                break  # ya se dibujó una vez
        
        
        if frame_count in peaks:
            current_exercise_data["angulo_rodilla"] = angle_knee
            current_exercise_data["angulo_cadera"] = angle_hip
            resultado = get_feedback(angle_knee, angle_hip, exercise)
            current_exercise_data["categoria"] = resultado["categoria"]
            current_exercise_data["feedback"] = resultado["feedback"]

            # Estructura limpia para cada sentadilla
            text_blocks.append({
                "sentadilla_index": sentadilla_index,
                "angulo_rodilla": round(current_exercise_data['angulo_rodilla'], 1),
                "angulo_cadera": round(current_exercise_data['angulo_cadera'], 1),
                "categoria": current_exercise_data['categoria'],
                "feedback": current_exercise_data['feedback']
            })

            # Reiniciar para la próxima sentadilla
            current_exercise_data = {
                "angulo_rodilla": None,
                "angulo_cadera": None,
                "categoria": None,
                "feedback": None,
            }
            sentadilla_index += 1

        if frame_count in start_frames or frame_count in end_frames or frame_count in peaks:
            for _ in range(int(fps * sec_pause)):
                out.write(frame)
        else:
            out.write(frame)

        frame_count += 1

    cap.release()
    out.release()

    ffmpeg.input(video_avi).output(video_mp4, vcodec="libx264", acodec="aac").overwrite_output().run()
    os.remove(video_avi)

    logger.info(f"Video procesado y guardado en {video_mp4}")
    return text_blocks
    
def get_feedback(angle_knee: float, angle_hip: float, exercise: str) -> dict:
    """
    Evalúa una sentadilla o deadlift en base a los ángulos internos de la rodilla y la cadera.

    Parámetros:
    - angle_knee: Ángulo interno de la rodilla (en grados).
    - angle_hip: Ángulo interno de la cadera (en grados).
    - exercise: String que indica el ejercicio ("squat" o "deadlift").

    Retorna:
    - dict con "categoria" (str) y "feedback" (str)
    """
    if exercise == "squat":
        return evaluate_squat(angle_knee, angle_hip)
    elif exercise == "deadlift":
        return evaluate_deadlift(angle_knee, angle_hip)
    else:
        raise ValueError("Ejercicio no reconocido. Debe ser 'squat' o 'deadlift'.")

def evaluate_squat(angle_knee: float, angle_hip: float) -> dict:
    """    Evalúa una sentadilla en base a los ángulos internos de la rodilla y la cadera.
    Parámetros:
    - angle_knee: Ángulo interno de la rodilla (en grados).
    - angle_hip: Ángulo interno de la cadera (en grados).
    Retorna:
    - dict con "categoria" (str) y "feedback" (str)
    """
    if angle_knee < 90 and 45 <= angle_hip <= 70:
        categoria = "✅ Correcta"
        feedback = "Buena profundidad y postura. Sentadilla realizada correctamente."

    elif 90 <= angle_knee <= 100 and 80 <= angle_hip <= 100:
        categoria = "🟡 Casi correcta"
        feedback = "Falta algo de profundidad. Mejorar movilidad de tobillo y control del descenso."

    elif 85 <= angle_knee <= 95 and 65 <= angle_hip <= 75:
        categoria = "🟡 Casi correcta"
        feedback = "La técnica es aceptable pero podría bajarse un poco más manteniendo el control."

    elif angle_knee < 70 and angle_hip > 90:
        categoria = "🟡 Casi correcta"
        feedback = "Exceso de flexión de cadera. Reforzar core y mantener el torso más erguido."

    elif angle_knee > 110 and angle_hip < 70:
        categoria = "⛔️ Incorrecta"
        feedback = "Exceso de flexión en rodillas. Riesgo articular elevado. Reajustar técnica y distribuir peso."

    elif angle_knee > 100 and angle_hip > 90:
        categoria = "⛔️ Incorrecta"
        feedback = "Sentadilla demasiado superficial. Aumentar profundidad para mayor activación muscular."

    elif angle_knee < 70 and angle_hip > 100:
        categoria = "⛔️ Incorrecta"
        feedback = "Inclinación excesiva del torso. Podría haber sobrecarga lumbar. Corregir patrón de movimiento."

    else:
        categoria = "Casi correcta"
        feedback = "La sentadilla no está mal, pero hay espacio para mejorar técnica y control postural."

    return {
        "categoria": categoria,
        "feedback": feedback
    }
    
def evaluate_deadlift(angle_knee: float, angle_hip: float) -> dict:
    """
    Evalúa un peso muerto en base a los ángulos internos de la rodilla y la cadera.

    Parámetros:
    - angle_knee: Ángulo interno de la rodilla (en grados).
    - angle_hip: Ángulo interno de la cadera (en grados).

    Retorna:
    - dict con "categoria" (str) y "feedback" (str)
    """

    if 100 <= angle_knee <= 130 and 60 <= angle_hip <= 90:
        categoria = "Correcta"
        feedback = "Buena postura inicial. La cadera está más alta que la rodilla, espalda lista para tirar correctamente."

    elif angle_knee < 90 and angle_hip > 90:
        categoria = "Incorrecta"
        feedback = "Parece más una sentadilla que un peso muerto. La cadera está demasiado baja."

    elif angle_knee > 130 and angle_hip < 60:
        categoria = "Incorrecta"
        feedback = "La cadera está demasiado alta y las piernas muy rectas. Riesgo de forzar la zona lumbar."

    elif 90 <= angle_knee <= 100 and 90 <= angle_hip <= 100:
        categoria = "Casi correcta"
        feedback = "Postura aceptable, pero hay poca dominancia de cadera. Elevar ligeramente la cadera y asegurar activación del core."

    elif angle_knee > 130 and angle_hip > 100:
        categoria = "Incorrecta"
        feedback = "Extensión inicial excesiva. Es posible que el tronco esté muy vertical. Requiere mayor inclinación de cadera."

    else:
        categoria = "Casi correcta"
        feedback = "La postura no es peligrosa, pero podría mejorarse la proporción entre cadera y rodilla para un patrón más eficiente."

    return {
        "categoria": categoria,
        "feedback": feedback
    }