import cv2
from predict import detect_objects

tracker = cv2.TrackerKCF_create()

data_output_path = "outputs/"
data_video_path = "evaluation/evaluation_data/videos/"
video_n = "video_3"
video = cv2.VideoCapture(f"{data_video_path}{video_n}.mp4")
ret, frame = video.read()

if not video.isOpened():
    print("No se pudo abrir el video")
    exit()
else:
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

video_output_file_name = data_output_path + video_n + "_tracker.avi"
video_out = cv2.VideoWriter(video_output_file_name, cv2.VideoWriter_fourcc(*'XVID'), 30, (height, width))
bounding_boxes = detect_objects(source=frame, conf=0.2, line_width=2, classes=0)
while len(bounding_boxes) == 0:
    print("No se detectaron objetos en el frame, intentando con el siguiente")
    ret, frame = video.read()
    bounding_boxes = detect_objects(source=frame, conf=0.2, line_width=2, classes=0)
    if not ret:
        print("No se detectaron objetos en el video")
        exit()
bbox = (bounding_boxes[0]["x"], bounding_boxes[0]["y"], bounding_boxes[0]["w"], bounding_boxes[0]["h"])

# Inicializa el tracker con la ROI seleccionada
tracker.init(frame, bbox)

# Loop para rastrear
while True:
    ret, frame = video.read()
    if not ret:
        break

    # Actualiza el tracker
    success, bbox = tracker.update(frame)

    if success:
        # Dibuja la caja delimitadora
        x, y, w, h = [int(i) for i in bbox]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2, 1)
    else:
        cv2.putText(frame, "Tracking perdido", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

    rotated_frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    
    video_out.write(rotated_frame)

video.release()
video_out.release()