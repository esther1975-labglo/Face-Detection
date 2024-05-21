import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
import csv

path = 'Images'
attendance_path = 'attendance'
os.makedirs(attendance_path, exist_ok=True)  

images = []
classNames = []

for person_name in os.listdir(path):
    person_path = os.path.join(path, person_name)
    if os.path.isdir(person_path):
        for img_name in os.listdir(person_path):
            curImg = cv2.imread(os.path.join(person_path, img_name))
            images.append(curImg)
            classNames.append(person_name)

def findEncodings(images, classNames):
    encodeList = []
    knownNames = []
    for img, name in zip(images, classNames):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        face_encodings = face_recognition.face_encodings(img)
        if len(face_encodings) > 0:
            encodeList.append(face_encodings[0])
            knownNames.append(name)
        else:
            print(f"No face detected in the image for {name}.")
    return encodeList, knownNames

def markAttendance(name):
    today = datetime.now().strftime('%Y-%m-%d')
    filename = os.path.join(attendance_path, f'Attendance_{today}.csv')

    if not os.path.exists(filename):
        with open(filename, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['Name', 'Time', 'Date'])
            writer.writeheader()
    
    with open(filename, 'r', newline='') as f:
        reader = csv.DictReader(f)
        nameList = [row['Name'] for row in reader]

    if name not in nameList:
        time_now = datetime.now()
        tString = time_now.strftime('%H:%M:%S')
        dString = time_now.strftime('%d/%m/%Y')
        with open(filename, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['Name', 'Time', 'Date'])
            writer.writerow({'Name': name, 'Time': tString, 'Date': dString})

encodeListKnown, knownNames = findEncodings(images, classNames)
print('Encoding Complete')

cap = cv2.VideoCapture(0)


cv2.namedWindow('CCTV Monitor', cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty('CCTV Monitor', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
MAX_DISTANCE = 100  
while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        y1, x2, y2, x1 = faceLoc
        box_height = abs(y2 - y1)

        if box_height > MAX_DISTANCE:
            continue

        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        print(faceDis)
        matchIndex = np.argmin(faceDis)
        if matches[matchIndex]:
            name = knownNames[matchIndex].upper()
            print(name)
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            y1, y2 = y1 - 50, y2 + 50  
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2-35), (x2, y2), (0, 250, 0), cv2.FILLED)
            cv2.putText(img, name, (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            markAttendance(name)
    
    cv2.imshow('CCTV Monitor', img)
    if cv2.waitKey(10) == 13:
        break

cap.release()
cv2.destroyAllWindows()
