"""Tiny AzkaVision"""
from ultralytics import YOLO
import numpy as np
import cv2
from testing.cropper import crop_cars
from FCC.CarClassifier import CarClassifier

model = YOLO("testing/yolov8s.pt")
model.to('cuda')

fcc = CarClassifier("testing/best.pt")

cap = cv2.VideoCapture('testing/MoA_cars.test.mp4')
output = cv2.VideoWriter("testing/output.mp4",cv2.VideoWriter_fourcc(*'MJPG'),30,(1080,1920))

while 1:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (1920, 1080))
    carBoxes = crop_cars(model, frame)
    for detections in carBoxes:
        for box in detections:
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (30, 255, 30), 2)
            carName = fcc.predict(frame, 0.2)
            cv2.putText(frame, carName, (box[0], box[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (30,255,30), 3)


    cv2.imshow("parking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    else:
        output.write(frame)

output.release()
cv2.destroyAllWindows()
cap.release()
