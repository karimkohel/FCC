import os, shutil

cars = os.listdir("dataset/train")

for car in cars:
    carPics = os.listdir(f"dataset/train/{car}/")
    carPics = carPics[:100]
    for carPic in carPics:
        shutil.move(f"dataset/train/{car}/{carPic}", f"dataset/val/{car}/{carPic}")