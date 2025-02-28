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
            z=point.z * w,
            visibility=point.visibility
        )
        scaled_landmarks.append(scaled_point)
    return scaled_landmarks

def draw_lines_and_angle(image, points_group, line_color=(0, 255, 0), angle_color=(0, 255, 255), text_color=(200, 200, 255)):
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

        # Dibujar las líneas BA y BC
        cv2.line(image, pointA, pointB, line_color, 2)  
        cv2.line(image, pointB, pointC, line_color, 2)  

        mean_z = (points[0].z + points[1].z + points[2].z) / 3
        if mean_z < max_z:
            max_z = mean_z
            id_max_z = idx

    points = points_group[id_max_z]
    # Extraer los puntos desde el diccionario
    pointA = (int(points[0].x), int(points[0].y))
    pointB = (int(points[1].x), int(points[1].y))
    pointC = (int(points[2].x), int(points[2].y))
    angle = get_angle(points)   
    # Resto del código permanece igual
    radius = 30  # Radio de la elipse
    axes = (radius, radius)  # Ejes de la elipse
    center = pointB  # Centro de la elipse en el punto B

    # Calcular los ángulos de inicio y fin
    start_angle = np.degrees(np.arctan2(pointA[1] - pointB[1], pointA[0] - pointB[0]))
    end_angle = np.degrees(np.arctan2(pointC[1] - pointB[1], pointC[0] - pointB[0]))

    # Calcular el ángulo de barrido más pequeño
    if end_angle - start_angle > 180:
        # El ángulo más pequeño es en sentido antihorario
        start_angle += 360

    # Determinar un ángulo intermedio (en el centro del barrido)
    middle_angle = np.radians((start_angle + end_angle) / 2)

    # Calcular la posición del texto a lo largo de la elipse
    text_x = int(center[0] + radius * 1.2 * np.cos(middle_angle))
    text_y = int(center[1] + radius * 1.2 * np.sin(middle_angle))
    text_position = (text_x, text_y)

    # Dibujar la elipse
    cv2.ellipse(image, center, axes, 0, start_angle, end_angle, angle_color, 2)


    # Dibujar el texto del ángulo en la trayectoria de la elipse
    cv2.putText(image, f"{angle:.2f}", text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)


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
    # Carga el modelo
    model = YOLO('yolo_best.pt')

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
        None
    """
    frame_numbers = []
    bbox_heights = []
    initial_frame = 0

    video_name = os.path.splitext(os.path.basename(video_path))[0]

    video_folder = os.path.join(output_path, video_name)
    os.makedirs(video_folder, exist_ok=True)

    video_avi = os.path.join(video_folder, "video_tracker.avi")
    video_mp4 = os.path.join(video_folder, "video_tracker.mp4")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: No se pudo abrir el video {video_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec para .avi

    if width > height:
        print("El video está en formato horizontal, se rotará 90° en sentido horario.")
        width, height = height, width
        rotate = True
    else:
        rotate = False
    
    out = cv2.VideoWriter(video_avi, fourcc, fps, (width, height))

    ret, frame = cap.read()
    if rotate:
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    bounding_boxes = detect_objects(source=frame, conf=0.2, line_width=2, classes=0)
    while len(bounding_boxes) == 0:
        ret, frame = cap.read()
        initial_frame += 1
        bounding_boxes = detect_objects(source=frame, conf=0.2, line_width=2, classes=0)
        if not ret:
            print("No se detectaron objetos en el video")
            exit()
    bbox = (bounding_boxes[0]["x"], bounding_boxes[0]["y"], bounding_boxes[0]["w"], bounding_boxes[0]["h"])
    tracker.init(frame, bbox)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if rotate:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

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

    print(f"Video procesado y guardado en {video_mp4}")
    return frame_numbers, bbox_heights, initial_frame

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

def detect_squat_repetitions(bbox_heights, frame_numbers, output_path, size=16,
                             distance=5, threshold=0.01, min_distance_from_peak=2):
    # 1. Suavizar la señal
    frame_numbers = np.array(frame_numbers)
    bbox_heights = np.array(bbox_heights)
    bbox_heights_smooth = uniform_filter1d(bbox_heights, size=size)
    
    # 2. Encontrar picos (máximos) y valles (mínimos)
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
    plt.plot(frame_numbers[peaks], bbox_heights_smooth[peaks], "x", label="Picos (máximos)", color='red')
    plt.plot(frame_numbers[valleys], bbox_heights_smooth[valleys], "o", label="Valles (mínimos)", color='green')
    
    for start, end in zip(start_frames, end_frames):
        plt.axvline(x=frame_numbers[start], color='orange', linestyle='--', label="Inicio repetición" if start == start_frames[0] else "")
        plt.axvline(x=frame_numbers[end], color='purple', linestyle='--', label="Fin repetición" if end == end_frames[0] else "")
    
    plt.title("Detección de repeticiones de sentadillas")
    plt.xlabel("Número de frame")
    plt.ylabel("Altura del bbox (y + h)")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{output_path}/rep_squat_graph.png")
    
    # # 5. Mostrar los fotogramas de inicio y fin de cada repetición
    # for i, (start, end) in enumerate(zip(start_frames, end_frames)):
    #     print(f"Repetición {i + 1}:")
    #     print(f"  - Inicio: Frame {frame_numbers[start]}")
    #     print(f"  - Pico: Frame {peaks[i]}")
    #     print(f"  - Fin:    Frame {frame_numbers[end]}")
    
    return start_frames, end_frames, peaks

def create_video(video_path, output_path, start_frames,
                 end_frames, peaks, sec_pause=3, initial_frame=0):
    
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    video_folder = os.path.join(output_path, video_name)
    os.makedirs(video_folder, exist_ok=True)

    video_avi = os.path.join(video_folder, "result.avi")
    video_mp4 = os.path.join(video_folder, "result.mp4")

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: No se pudo abrir el video {video_path}")
        return
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec para .avi
    
    if width > height:
        print("El video está en formato horizontal, se rotará 90° en sentido horario.")
        width, height = height, width
        rotate = True
    else:
        rotate = False
        
    out = cv2.VideoWriter(video_avi, fourcc, fps, (width, height))
    for _ in range(initial_frame+1):
        ret, frame = cap.read()
    if rotate:
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if rotate:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        
        if (frame_count in start_frames) or (frame_count in end_frames) or (frame_count in peaks):
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb_frame)
            # Dibujar las poses detectadas en el frame
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
                if frame_count in start_frames:
                    cv2.putText(frame, "Inicio de sentadilla", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                if frame_count in end_frames:
                    cv2.putText(frame, "Fin de sentadilla", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                if frame_count in peaks:
                    cv2.putText(frame, "Punto minimo", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

                draw_lines_and_angle(frame, (right_knee_points,left_knee_points))    
                draw_lines_and_angle(frame, (right_hip_points, left_hip_points))
                for _ in range(int(fps*sec_pause)):
                    out.write(frame)
                    
    
        # Escribir el frame procesado en el video de salida
        out.write(frame)
        frame_count += 1

    # Liberar recursos
    cap.release()
    out.release()
    
    ffmpeg.input(video_avi).output(video_mp4, vcodec="libx264", acodec="aac").overwrite_output().run()
    # Borra el archivo .avi 
    os.remove(video_avi)   

    print(f"Video procesado y guardado en {video_mp4}")