from ultralytics import YOLO
import cv2

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
