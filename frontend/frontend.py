import streamlit as st
import requests
import time
from pathlib import Path
import os
import base64 
import logging

# Configurar logging
logging.basicConfig(
    level=logging.INFO,  # Nivel de logging (INFO, WARNING, ERROR, DEBUG)
    handlers=[logging.StreamHandler()]  # Mostrar logs en la consola
)
logger = logging.getLogger(__name__)

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.title("🏋️‍♂️ Gym Tracker - Procesamiento de Videos")

# Subida de video
st.header("1. Subir Video")
uploaded_file = st.file_uploader("Selecciona un archivo .mp4", type=["mp4"])

if uploaded_file is not None:
    filename = uploaded_file.name

    # Subir el archivo al backend
    with st.spinner("Subiendo video..."):
        files = {"file": (uploaded_file.name, uploaded_file, "video/mp4")}
        response = requests.post(f"{API_URL}/upload-video", files=files)
        
    if response.status_code == 200:
        st.success("✅ Video subido exitosamente")
        st.session_state["uploaded_filename"] = filename
    else:
        st.error("❌ Error al subir el video")

# Seleccionar tipo de ejercicio
if "uploaded_filename" in st.session_state:
    st.header("2. Selecciona tipo de ejercicio")
    exercise_type = st.selectbox("Tipo de ejercicio", ["squat"]) # , "deadlift"])
    if st.button("Procesar video"):
        with st.spinner("Procesando video..."):
            response = requests.post(
                f"{API_URL}/process-{exercise_type}",
                params={"filename": st.session_state["uploaded_filename"]}
            )
        if response.status_code == 200:
            result = response.json()
            st.success("✅ Video procesado exitosamente")
            st.session_state["processed"] = True
            st.session_state["text_blocks"] = result.get("text_blocks", [])
        else:
            st.error("❌ Error al procesar el video")

# Mostrar video procesado y gráfico
if st.session_state.get("processed"):
    st.header("3. Resultado del procesamiento")

    video_name = Path(st.session_state["uploaded_filename"]).stem

    # Mostrar video procesado
    st.subheader("📹 Video procesado")

    col1, col2 = st.columns([3, 2])  # Ajustás el ancho relativo entre video y texto

    with col1:
        video_response = requests.get(f"{API_URL}/get-video", params={"video_name": video_name})
        if video_response.status_code == 200:
            video_bytes = video_response.content
            b64_video = base64.b64encode(video_bytes).decode("utf-8")

            video_html = f"""
                <style>
                .responsive-video {{
                    max-height: 90vh;
                    max-width: 100%;
                    width: auto;
                    height: auto;
                    display: block;
                    margin-left: auto;
                    margin-right: auto;
                }}
                </style>

                <video class="responsive-video" controls>
                    <source src="data:video/mp4;base64,{b64_video}" type="video/mp4">
                    Tu navegador no soporta video.
                </video>
            """
            st.markdown(video_html, unsafe_allow_html=True)
        else:
            st.error("No se pudo obtener el video procesado.")

    with col2:
        st.markdown("### 📝 Detalles del análisis")

        bloques = st.session_state.get("text_blocks", [])
        if bloques:
            scrollable_html = "<div style='max-height:70vh; overflow-y:auto; padding-right:10px;'>"
            for bloque in bloques:
                category = bloque['categoria']
                if "Correcta" in category:
                    color = "#32CD32"
                elif "Casi correcta" in category:
                    color = "#DAA520"
                else:
                    color = "#FF6347"
                    
                scrollable_html = "\n".join([
                    scrollable_html,
                    f"<h5>🏋️ Sentadilla {bloque['sentadilla_index']}</h5>",
                    f"<p style='font-size:13px; margin-left:10px;'>• <b>Ángulo rodilla:</b> {bloque['angulo_rodilla']:.1f}°</p>",
                    f"<p style='font-size:13px; margin-left:10px;'>• <b>Ángulo cadera:</b> {bloque['angulo_cadera']:.1f}°</p>",
                    f"<p style='font-size:13px; margin-left:10px;'>• <b>Categoría:</b> <span style='color:{color}; font-weight:600;'>{category}</span></p>",
                    f"<p style='font-size:12px; margin-left:10px; color:#ccc;'><i>{bloque['feedback']}</i></p>",
                    "<hr style='margin:5px 0;'>"
                ])
            scrollable_html += "</div>"
            st.markdown(scrollable_html, unsafe_allow_html=True)
        else:
            st.info("No se encontraron mensajes de análisis.")

    # Mostrar gráfico
    st.subheader("📈 Gráfico de repeticiones")
    graph_response = requests.get(f"{API_URL}/get-graph", params={"video_name": video_name})

    if graph_response.status_code == 200:
        st.image(graph_response.content, caption="Gráfico de repeticiones")
    else:
        st.error("No se pudo obtener el gráfico.")