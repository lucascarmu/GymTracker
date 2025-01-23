import cv2
from ultralytics import YOLO

tracker_types = ['KCF']
data_output_path = "evaluation_data/outputs/"
data_video_path = "evaluation_data/videos/"

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
    model = YOLO('../yolo_best.pt')

    # Realiza la predicción
    results = model.predict(source=source, conf=conf, save=False, line_width=line_width)

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

def calculate_iou(boxA, boxB):
        """
        Calcula el Intersection over Union (IoU) entre dos cajas delimitadoras.
        Las cajas están en formato (x, y, w, h).
        """
        # Coordenadas de las esquinas de las cajas
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
        yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

        # Calcula el área de intersección
        interWidth = max(0, xB - xA)
        interHeight = max(0, yB - yA)
        interArea = interWidth * interHeight

        # Calcula el área de unión
        boxAArea = boxA[2] * boxA[3]
        boxBArea = boxB[2] * boxB[3]
        unionArea = boxAArea + boxBArea - interArea

        # Calcula IoU
        if unionArea == 0:
            return 0  # Evitar división por cero
        return interArea / unionArea

def evaluate_video(video_name, tracker_type):
    print(f"Evaluando video {video_name} con tracker {tracker_type}")
    if tracker_type == 'BOOSTING':
        tracker = cv2.legacy_TrackerBoosting.create()
    elif tracker_type == 'MIL':
        tracker = cv2.TrackerMIL_create()
    elif tracker_type == 'KCF':
        tracker = cv2.TrackerKCF_create()
    elif tracker_type == 'TLD':
        tracker = cv2.legacy_TrackerTLD.create()
    elif tracker_type == 'MEDIANFLOW':
        tracker = cv2.legacy_TrackerMedianFlow.create()
    elif tracker_type == 'MOSSE':
        tracker = cv2.legacy.TrackerMOSSE_create()
    elif tracker_type == 'CSRT':
        tracker = cv2.legacy.TrackerCSRT_create()
    else:
        tracker = None
        print('Tracker no encontrado')
    
    video = cv2.VideoCapture(f"{data_video_path}{video_name}.mp4")
    ret, frame = video.read()

    if not video.isOpened():
        print("No se pudo abrir el video")
        exit()
    else:
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    video_output_file_name = f"{data_output_path}{video_name}/{tracker_type}.avi"
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

    # Dentro del bucle principal:
    total_iou = 0
    frame_count = 0

    while True:
        ret, frame = video.read()
        if not ret:
            break

        bounding_boxes = detect_objects(source=frame, conf=0.2, line_width=2, classes=0)
        if len(bounding_boxes) > 0:
            # Bounding box de la predicción
            pred_bbox = (bounding_boxes[0]["x"], bounding_boxes[0]["y"], bounding_boxes[0]["w"], bounding_boxes[0]["h"])
            cv2.rectangle(frame, (pred_bbox[0], pred_bbox[1]), 
                        (pred_bbox[0] + pred_bbox[2], pred_bbox[1] + pred_bbox[3]), (0, 255, 0), 2, 1)
        else:
            continue  # No hay predicción, saltar este frame

        # Actualiza el tracker
        success, track_bbox = tracker.update(frame)

        if success:
            x, y, w, h = [int(i) for i in track_bbox]
            # Bounding box del tracker
            track_bbox = (x, y, w, h)

            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2, 1)

            # Calcula IoU y acumula
            iou = calculate_iou(pred_bbox, track_bbox)
            total_iou += iou
            frame_count += 1

            # Muestra IoU en el frame
            cv2.putText(frame, f"IoU: {iou:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        else:
            cv2.putText(frame, "Tracking perdido", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

        # Escribe el frame con el video rotado
        rotated_frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        video_out.write(rotated_frame)

    video.release()
    video_out.release()

    # Calcula la precisión general
    if frame_count > 0:
        average_iou = total_iou / frame_count
        print(f"Precisión promedio del tracker (IoU): {average_iou:.2f}")
        with open(f"{data_output_path}{video_name}/results.txt", "a") as file:
            file.write(f"{tracker_type}: {average_iou:.2f}\n")
    else:
        print("No se calculó IoU debido a la falta de frames con detecciones válidas.") 


if __name__ == "__main__":
    for tracker_type in tracker_types:
        evaluate_video("video_1", tracker_type)
        #evaluate_video("video_3", tracker_type)