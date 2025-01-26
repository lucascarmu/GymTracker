import cv2
import mediapipe as mp
from utils import plot_3d_landmarks, scale_landmarks, draw_lines_and_angle


# Inicializar MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Ruta de la imagen de entrada
input_image_path = "images/squat.jpg"  # Cambiar a la ruta de tu imagen
output_image_path = "images/output_image.jpg"  # Ruta donde se guardará la imagen procesada

# Cargar la imagen
image = cv2.imread(input_image_path)
if image is None:
    print("No se pudo cargar la imagen. Verifica la ruta.")
    exit()

# Convertir a RGB (MediaPipe requiere RGB)
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Procesar la imagen con MediaPipe Pose
results = pose.process(rgb_image)

# Verificar si se detectaron puntos clave
if results.pose_landmarks:
    h, w, _ = image.shape  # Altura y ancho de la imagen

    right_knee_points = scale_landmarks(
        [results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP.value],
            results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE.value],
            results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE.value]],
        w, h
    )
    left_knee_points = scale_landmarks(
        [results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP.value],
            results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE.value],
            results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE.value]],
        w, h
    )
    right_hip_points = scale_landmarks(
        [results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
            results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP.value],
            results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE.value]],
        w, h
    )
    left_hip_points = scale_landmarks(
        [results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
            results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP.value],
            results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE.value]],
        w, h
    )

    draw_lines_and_angle(image, (right_knee_points,left_knee_points))    
    draw_lines_and_angle(image, (right_hip_points, left_hip_points))
    
# Guardar la imagen procesada
cv2.imwrite(output_image_path, image)
print(f"Imagen procesada guardada en: {output_image_path}")

# Liberar recursos
pose.close()
