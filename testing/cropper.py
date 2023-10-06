from ultralytics import YOLO
import numpy as np
import cv2

def crop_cars(mode, frame: np.ndarray) -> None:
    results = model.predict(source=frame, conf=0.1, stream=True)
    for result in results:
        for predictedClass in result.boxes.cls:
            if int(predictedClass) in [2, 7]:
                boxes = result.boxes.xyxy.to('cpu').numpy().astype(int)
                confidences = result.boxes.conf.to('cpu').numpy().astype(float)

                for box, conf in zip(boxes, confidences):
                    x_min, y_min, x_max, y_max = box
                    cropped = frame[y_min:y_max, x_min:x_max]


if __name__ == "__main__":
    model = YOLO("testing/yolov8s.pt")
    model.to('cuda')

    frame = cv2.imread("testing/car.test.jpg")
    crop_cars(model, frame)
