from ultralytics import YOLO
import cv2

class PersonDetector:
    def __init__(self, model_path="yolov8n.pt"):
        self.model = YOLO(model_path)

    def detect(self, frame):
        results = self.model(frame, verbose=False)
        persons = []
        for r in results:
            for box in r.boxes:
                if int(box.cls[0]) == 0 and float(box.conf[0]) > 0.5:
                    persons.append(box)
        return persons

    def draw_boxes(self, frame, persons):
        for box in persons:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
            cv2.putText(frame, f"Personne {conf:.0%}", (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 2)
        return frame