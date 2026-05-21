import cv2

class FaceDetector:
    def __init__(self):
        cascade = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.detector = cv2.CascadeClassifier(cascade)

    def detect(self, frame):
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))
        return len(faces) > 0, [(y, x+w, y+h, x) for (x, y, w, h) in faces]

    def draw(self, frame, faces):
        for (top, right, bottom, left) in faces:
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 220, 0), 2)
            cv2.rectangle(frame, (left, bottom-28), (right, bottom),
                          (0, 220, 0), cv2.FILLED)
            cv2.putText(frame, "Visage detecte", (left+4, bottom-7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        return frame