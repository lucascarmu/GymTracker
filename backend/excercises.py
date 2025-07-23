import os
import cv2
import logging
from utils import analize_bbox, detect_squat_repetitions, create_video, detect_deadlift_repetitions

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def process_squat_video(input_path: str, output_path: str):
    """
    Función para procesar un video: analiza las repeticiones de sentadillas y guarda el video procesado.
    """
    logger.info(f"Procesando video: {input_path}")

    # Inicializar el tracker
    tracker = cv2.legacy.TrackerBoosting_create()

    # Analizar el bbox y obtener datos
    logger.info("Analizando bounding box...")
    frame_numbers, bbox_heights, initial_frame, bboxes = analize_bbox(
        video_path=input_path,
        output_path=output_path,
        tracker=tracker
    )

    # Detectar repeticiones de sentadillas
    logger.info("Detectando repeticiones de sentadillas...")
    start_frames, end_frames, peaks = detect_squat_repetitions(
        bbox_heights=bbox_heights,
        frame_numbers=frame_numbers,
        output_path=f"{output_path}/{os.path.splitext(os.path.basename(input_path))[0]}"
    )

    # Crear el video procesado
    logger.info("Creando video procesado...")
    text_blocks = create_video(
        video_path=input_path,
        output_path=output_path,
        exercise="squat",
        start_frames=start_frames,
        end_frames=end_frames,
        peaks=peaks,
        bboxes=bboxes,
        sec_pause=1.5,
        initial_frame=initial_frame
    )

    for line in text_blocks:
        logger.info(line)

    logger.info(f"Video procesado guardado en: {output_path}")
    return text_blocks

def process_deadlift_video(input_path: str, output_path: str):
    """
    Función para procesar un video: analiza las repeticiones de peso muerto y guarda el video procesado.
    """
    logger.info(f"Procesando video de peso muerto: {input_path}")

    # Inicializar el tracker
    tracker = cv2.legacy.TrackerBoosting_create()

    # Analizar el bbox y obtener datos
    logger.info("Analizando bounding box...")
    frame_numbers, bbox_heights, initial_frame, bboxes = analize_bbox(
        video_path=input_path,
        output_path=output_path,
        tracker=tracker
    )

    # Detectar repeticiones de peso muerto
    logger.info("Detectando repeticiones de peso muerto...")
    start_frames, end_frames, peaks = detect_deadlift_repetitions(
        bbox_heights=bbox_heights,
        frame_numbers=frame_numbers,
        output_path=f"{output_path}{os.path.splitext(os.path.basename(input_path))[0]}"
    )

    # Crear el video procesado
    logger.info("Creando video procesado...")
    text_blocks = create_video(
        video_path=input_path,
        output_path=output_path,
        exercise="deadlift",
        start_frames=start_frames,
        end_frames=end_frames,
        peaks=peaks,
        bboxes=bboxes,
        sec_pause=1.5,
        initial_frame=initial_frame
    )
    
    for line in text_blocks:
        logger.info(line)

    logger.info(f"Video procesado guardado en: {output_path}")
    return text_blocks

