import face_recognition
import cv2
import numpy as np
import csv
import os
from datetime import datetime

video_capture = cv2.VideoCapture(0)
input_active = True  # Flag to control input taking

sipun_image = face_recognition.load_image_file("photos/sipun.jpg")
sipun_encoding = face_recognition.face_encodings(sipun_image)[0]

deepak_image = face_recognition.load_image_file("photos/deepak.jpg")
deepak_encoding = face_recognition.face_encodings(deepak_image)[0]

unknown_image = face_recognition.load_image_file("photos/unknown.jpg")
unknown_encoding = face_recognition.face_encodings(unknown_image)[0]

sujeet_image = face_recognition.load_image_file("photos/sujeet.jpg")
sujeet_encoding = face_recognition.face_encodings(sujeet_image)[0]

dipanshu_image = face_recognition.load_image_file("photos/dipanshu.jpg")
dipanshu_encoding = face_recognition.face_encodings(dipanshu_image)[0]

known_face_encoding = [
    sipun_encoding,
    deepak_encoding,
    unknown_encoding,
    sujeet_encoding,
    dipanshu_encoding
]

known_face_names = [
    "Sipun",
    "Deepak",
    "Niki",
    "Sujeet",
    "Dipanshu"
]

students = known_face_names.copy()

face_locations = []
face_encodings = []
face_names = []
now = datetime.now()
current_date = now.strftime("%Y-%m-%d")

f = open(current_date + ".csv", 'a', newline='')
lnwriter = csv.writer(f)

while True:
    _, frame = video_capture.read()
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    
    if input_active:  # Check if input is active
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
                    print(students)
                    current_time = now.strftime("%H:%M:%S")
                    lnwriter.writerow([name, current_time, current_date])
    
    cv2.imshow("Attendance System", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
f.close()
