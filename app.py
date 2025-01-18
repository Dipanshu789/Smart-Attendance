from flask import Flask, render_template, Response, send_file
import cv2
import face_recognition
import numpy as np
import csv
import os
from datetime import datetime
from threading import Thread

app = Flask(__name__, static_url_path='/static')

video_capture = cv2.VideoCapture(0)

def load_known_faces():
    known_face_encodings = []
    known_face_names = []
    
    for name in ["sipun", "deepak", "unknown", "sujeet", "dipanshu"]:
        image = face_recognition.load_image_file(f"photos/{name}.jpg")
        encoding = face_recognition.face_encodings(image)[0]
        known_face_encodings.append(encoding)
        known_face_names.append(name.capitalize())
    
    return known_face_encodings, known_face_names

known_face_encoding, known_face_names = load_known_faces()
students = known_face_names.copy()

def generate_frames():
    while True:
        now = datetime.now()
        current_date = now.strftime("%Y-%m-%d")
        f = open(current_date + ".csv", 'a', newline='') 
        lnwriter = csv.writer(f)
        
        success, frame = video_capture.read()
        if not success:
            break
    
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        face_names = []

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encoding, face_encoding)
            name = ""
            face_distance = face_recognition.face_distance(known_face_encoding, face_encoding)
            best_match_index = np.argmin(face_distance)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)

            if name in known_face_names:
                if name in students:
                    students.remove(name)
                    current_time = now.strftime("%H:%M:%S")
                    lnwriter.writerow([name, current_time, current_date])

        cv2.imshow("Attendance System", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()
    f.close()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start')
def start_attendance():
    global input_active
    input_active = True 
    thread = Thread(target=generate_frames)
    thread.daemon = True
    thread.start()
    return "Attendance system started!"

@app.route('/download')
def download_data():
    now = datetime.now()
    current_date = now.strftime("%Y-%m-%d")
    return send_file(f"{current_date}.csv", as_attachment=True)

@app.route('/close')
def close_system():
    global input_active
    input_active = False  # Set the flag to stop input taking
    video_capture.release()
    cv2.destroyAllWindows()
    return "System input stopped!"

if __name__ == '__main__':
    app.run(debug=True)