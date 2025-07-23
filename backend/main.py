from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
import os
import logging
from excercises import process_squat_video, process_deadlift_video

# Configurar logging
logging.basicConfig(
    level=logging.INFO,  # Nivel de logging (INFO, WARNING, ERROR, DEBUG)
    handlers=[logging.StreamHandler()]  # Mostrar logs en la consola
)
logger = logging.getLogger(__name__)

# Crear una instancia de FastAPI
app = FastAPI()

# Configurar las rutas de subida y procesamiento. Si no se especifican, generar un error.
UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER")
PROCESSED_FOLDER = os.getenv("PROCESSED_FOLDER")
if not UPLOAD_FOLDER or not PROCESSED_FOLDER:
    raise ValueError("Las variables de entorno UPLOAD_FOLDER y PROCESSED_FOLDER deben estar definidas.")

@app.get("/")
def read_root():
    """
    Endpoint de prueba para verificar que la API está funcionando.
    """
    return {"message": "API de GymTracker está funcionando correctamente."}

@app.post("/process-squat")
def process_squat(filename: str):
    """
    Endpoint para procesar un video de sentadillas.
    """
    return process_video_generic(filename, "squat")


@app.post("/process-deadlift")
def process_deadlift(filename: str):
    """
    Endpoint para procesar un video de peso muerto.
    """
    return process_video_generic(filename, "deadlift")


def process_video_generic(filename: str, exercise_type: str):
    """
    Función genérica para procesar un video según el tipo de ejercicio.
    """
    logger.info(f"Procesando video de {exercise_type}: {filename}")

    # Ruta completa del archivo subido
    upload_path = os.path.join(UPLOAD_FOLDER, filename)

    # Verificar que el archivo exista
    if not os.path.exists(upload_path):
        logger.error(f"Archivo no encontrado: {filename}")
        raise HTTPException(status_code=404, detail="Archivo no encontrado")

    # Ruta completa donde se guardará el archivo procesado
    processed_path = os.path.join(PROCESSED_FOLDER)

    # Crear la carpeta de salida si no existe
    os.makedirs(processed_path, exist_ok=True)

    # Procesar el video
    logger.info(f"Iniciando procesamiento del video para {exercise_type}...")
    if exercise_type == "squat":
        text_blocks = process_squat_video(upload_path, processed_path)
    elif exercise_type == "deadlift":
        text_blocks = process_deadlift_video(upload_path, processed_path)
    else:
        logger.error(f"Tipo de ejercicio no soportado: {exercise_type}")
        raise HTTPException(status_code=400, detail="Tipo de ejercicio no soportado")

    logger.info(f"Video {filename} procesado exitosamente para {exercise_type}")
    return {"filename": filename,
            "exercise": exercise_type, 
            "message": "Video procesado exitosamente",
            "text_blocks": text_blocks}

@app.post("/upload-video")
async def upload_video(file: UploadFile = File(...)):
    """
    Endpoint para subir un archivo de video y guardarlo en la carpeta "uploads".
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

    logger.info(f"Archivo {file.filename} subido exitosamente")
    return {"filename": file.filename, "message": "Video subido exitosamente"}

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
        logger.error(f"Video no encontrado: {video_name} en {file_path}")
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