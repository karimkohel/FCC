from ultralytics import YOLO
import numpy as np
import cv2


def crop_cars(model, frame: np.ndarray) -> None:
    results = model.predict(source=frame, conf=0.7, stream=True)
    carsBoxes = []
    for result in results:
        for predictedClass in result.boxes.cls:
            if int(predictedClass) in [2, 7]:
                boxes = result.boxes.xyxy.to('cpu').numpy().astype(int)
                confidences = result.boxes.conf.to('cpu').numpy().astype(float)
                carsBoxes.append(boxes)
    return carsBoxes

                # for box, conf in zip(boxes, confidences):
                #     x_min, y_min, x_max, y_max = box
                #     cropped = frame[y_min:y_max, x_min:x_max]
                #     cv2.imshow("Cropped frame", cropped)
                #     cv2.waitKey(2000)


if __name__ == "__main__":
    model = YOLO("testing/yolov8s.pt")
    model.to('cuda')

    cap = cv2.VideoCapture('testing/night_arkan1.test.mp4')

    while 1:
        ret, frame = cap.read()
        carBoxes = crop_cars(model, frame)
        for detections in carBoxes:
            for box in detections:
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (30, 255, 30), 2)

        cv2.imshow("parking", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cv2.destroyAllWindows()
    cap.release()