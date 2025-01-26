import cv2
import mediapipe as mp
from utils import scale_landmarks, draw_lines_and_angle

def process_video(input_path, output_path):
    # Inicializar MediaPipe Pose
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()

    # Leer el video de entrada
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: No se pudo abrir el video {input_path}")
        return

    # Obtener las propiedades del video
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec para .avi

    # Configurar el video de salida
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convertir la imagen a RGB (MediaPipe trabaja en RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Realizar la detección de pose
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

            draw_lines_and_angle(frame, (right_knee_points,left_knee_points))    
            draw_lines_and_angle(frame, (right_hip_points, left_hip_points))

        # Escribir el frame procesado en el video de salida
        out.write(frame)

    # Liberar recursos
    cap.release()
    out.release()

    print(f"Video procesado y guardado en {output_path}")

# Uso del script
input_video = "evaluation/evaluation_data/videos/video_1.mp4"  # Path del video de entrada
output_video = "outputs/pose_detection/video_1_pose.avi"  # Path del video de salida
process_video(input_video, output_video)
