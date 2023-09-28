from ultralytics import YOLO


model = YOLO("train/yolov8m-cls.pt")

if __name__ == "__main__":
    results = model.train(data="dataset/", epochs=10, amp=False, batch=2, plots=True)