from flask import Flask, render_template, Response, jsonify
import cv2, time, threading, os

from database         import get_db, init_db
from detection.person import PersonDetector
from detection.face   import FaceDetector
from detection.main   import HandDetector
from logic.presence   import mark_attendance, get_present_today, get_total_today

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

init_db()
conn = get_db()

person_detector = PersonDetector(os.path.join(BASE_DIR, "yolov8n.pt"))
face_detector   = FaceDetector()
hand_detector   = HandDetector()

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

last_pointage = 0
person_count  = 0
last_message  = ""
lock          = threading.Lock()
COOLDOWN      = 10   # secondes entre deux pointages


#Pointage 

def generate_pointage():
    global person_count, last_message, last_pointage

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame   = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        #Detection personne (YOLO)
        persons = person_detector.detect(frame)
        frame   = person_detector.draw_boxes(frame, persons)
        nb      = len(persons)
        with lock:
            person_count = nb

        cv2.rectangle(frame, (0, 0), (230, 46), (0, 0, 0), -1)
        cv2.putText(frame, f"Personnes: {nb}", (8, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        if nb > 0:
            #Detection visage 
            face_ok, faces = face_detector.detect(frame)
            frame = face_detector.draw(frame, faces)

            #Detection main levee 
            main_levee, frame = hand_detector.hand_up(frame)

            #Pointage
            if face_ok and main_levee:
                now   = time.time()
                with lock:
                    depuis = now - last_pointage

                if depuis > COOLDOWN:
                    nom = mark_attendance(conn)
                    with lock:
                        last_pointage = now
                        last_message  = f"{nom} — PRÉSENT !"
                    cv2.rectangle(frame, (0, h-65), (w, h), (0, 140, 0), -1)
                    cv2.putText(frame, f"{nom} POINTE !",
                                (15, h-20), cv2.FONT_HERSHEY_SIMPLEX,
                                1.0, (255, 255, 255), 2)
                else:
                    restant = int(COOLDOWN - depuis) + 1
                    cv2.rectangle(frame, (0, h-65), (w, h), (40, 40, 40), -1)
                    cv2.putText(frame, f"Patientez {restant}s...",
                                (15, h-20), cv2.FONT_HERSHEY_SIMPLEX,
                                0.9, (0, 200, 255), 2)

            elif face_ok and not main_levee:
                cv2.rectangle(frame, (0, h-65), (w, h), (20, 20, 20), -1)
                cv2.putText(frame, "Visage detecte — Levez la main !",
                            (15, h-20), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 180, 255), 2)

        ret, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ret:
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
                   + buf.tobytes() + b'\r\n')


#Comptage

def generate_comptage():
    while True:
        success, frame = cap.read()
        if not success:
            break

        frame   = cv2.flip(frame, 1)
        persons = person_detector.detect(frame)
        frame   = person_detector.draw_boxes(frame, persons)
        nb      = len(persons)
        with lock:
            globals()['person_count'] = nb

        h, w, _ = frame.shape
        cv2.rectangle(frame, (0, 0), (w, 72), (15, 15, 15), -1)
        cv2.putText(frame, f"Personnes dans la salle : {nb}",
                    (16, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 150), 3)

        ret, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ret:
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
                   + buf.tobytes() + b'\r\n')


#Routes

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video_pointage")
def video_pointage():
    return Response(generate_pointage(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/comptage")
def comptage():
    return render_template("comptage.html")

@app.route("/video_comptage")
def video_comptage():
    return Response(generate_comptage(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/api/count")
def api_count():
    with lock:
        nb = person_count
    return jsonify({"count": nb})

@app.route("/api/message")
def api_message():
    with lock:
        msg = last_message
    return jsonify({"message": msg})

@app.route("/stats")
def stats():
    presents = get_present_today(conn)
    total    = get_total_today(conn)
    return render_template("stats.html", presents=presents, total=total)


if __name__ == "__main__":
    app.run(debug=False, threaded=True)