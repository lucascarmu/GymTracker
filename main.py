from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
import os
import cv2
import logging
from utils import analize_bbox, detect_squat_repetitions, create_video

# Configurar logging
logging.basicConfig(
    level=logging.INFO,  # Nivel de logging (INFO, WARNING, ERROR, DEBUG)
    format="%(asctime)s - %(levelname)s - %(message)s",  # Formato del mensaje
    handlers=[logging.StreamHandler()]  # Mostrar logs en la consola
)
logger = logging.getLogger(__name__)

# Crear una instancia de FastAPI
app = FastAPI()

# Crear las carpetas "uploads" y "processed" si no existen
UPLOAD_FOLDER = "uploads"
PROCESSED_FOLDER = "processed"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

def process_video(input_path: str, output_path: str):
    """
    Función para procesar un video: analiza las repeticiones de sentadillas y guarda el video procesado.
    """
    logger.info(f"Procesando video: {input_path}")

    # Inicializar el tracker
    tracker = cv2.legacy.TrackerBoosting_create()

    # Analizar el bbox y obtener datos
    logger.info("Analizando bounding box...")
    frame_numbers, bbox_heights, initial_frame = analize_bbox(
        video_path=input_path,
        output_path=output_path,
        tracker=tracker
    )

    # Detectar repeticiones de sentadillas
    logger.info("Detectando repeticiones de sentadillas...")
    start_frames, end_frames, peaks = detect_squat_repetitions(
        bbox_heights=bbox_heights,
        frame_numbers=frame_numbers,
        output_path=f"{output_path}{os.path.splitext(os.path.basename(input_path))[0]}"
    )

    # Crear el video procesado
    logger.info("Creando video procesado...")
    create_video(
        video_path=input_path,
        output_path=output_path,
        start_frames=start_frames,
        end_frames=end_frames,
        peaks=peaks,
        sec_pause=3,  # Pausa de 3 segundos en los puntos clave
        initial_frame=initial_frame
    )

    logger.info(f"Video procesado guardado en: {output_path}")

@app.post("/upload-video")
async def upload_video(file: UploadFile = File(...)):
    """
    Endpoint para subir un archivo de video, procesarlo y guardarlo en la carpeta "processed".
    """
    logger.info(f"Recibiendo archivo: {file.filename}")

    # Verificar que el archivo sea un video (opcional)
    if not file.filename.endswith(".mp4"):
        logger.error(f"Archivo no válido: {file.filename}")
        raise HTTPException(status_code=400, detail="Solo se permiten archivos .mp4")

    # Ruta completa donde se guardará el archivo subido
    upload_path = os.path.join(UPLOAD_FOLDER, file.filename)

    # Guardar el archivo en la carpeta "uploads"
    logger.info(f"Guardando archivo en: {upload_path}")
    with open(upload_path, "wb") as buffer:
        buffer.write(await file.read())

    # Ruta completa donde se guardará el archivo procesado
    processed_path = f"{PROCESSED_FOLDER}/"

    # Procesar el video (analizar repeticiones de sentadillas)
    logger.info("Iniciando procesamiento del video...")
    process_video(upload_path, processed_path)

    logger.info(f"Video {file.filename} procesado exitosamente")
    return {"filename": file.filename, "message": "Video subido y procesado exitosamente"}

@app.get("/get-video")
async def get_video(video_name: str):
    """
    Endpoint para devolver un video procesado por su nombre.
    """
    logger.info(f"Solicitando video: {video_name}")

    # Ruta completa del video procesado
    file_path = os.path.join(PROCESSED_FOLDER, video_name, "result.mp4")

    # Verificar si el archivo existe
    if not os.path.exists(file_path):
        logger.error(f"Video no encontrado: {video_name}")
        raise HTTPException(status_code=404, detail="Video no encontrado")

    # Devolver el archivo de video
    logger.info(f"Enviando video: {video_name}")
    return FileResponse(file_path, media_type="video/mp4")

@app.get("/get-graph")
async def get_graph(video_name: str):
    """
    Endpoint para devolver la gráfica de repeticiones de sentadillas por el nombre del video.
    """
    logger.info(f"Solicitando gráfica para el video: {video_name}")

    # Ruta completa de la gráfica
    graph_path = os.path.join(PROCESSED_FOLDER, video_name, "rep_squat_graph.png")

    # Verificar si la gráfica existe
    if not os.path.exists(graph_path):
        logger.error(f"Gráfica no encontrada para el video: {video_name}")
        raise HTTPException(status_code=404, detail="Gráfica no encontrada")

    # Devolver la gráfica como una imagen
    logger.info(f"Enviando gráfica: {graph_path}")
    return FileResponse(graph_path, media_type="image/png")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)