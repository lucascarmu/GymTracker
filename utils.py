import cv2
import mediapipe as mp
import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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

def draw_lines_and_angle(image, points_group):
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
        cv2.line(image, pointA, pointB, (0, 255, 0), 2)  # Línea verde para BA
        cv2.line(image, pointB, pointC, (255, 0, 0), 2)  # Línea azul para BC

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
    cv2.ellipse(image, center, axes, 0, start_angle, end_angle, (0, 255, 255), 2)  # Color amarillo


    # Dibujar el texto del ángulo en la trayectoria de la elipse
    cv2.putText(image, f"{angle:.2f}", text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 255), 2)


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