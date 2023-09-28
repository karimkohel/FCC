"""This was a handy fast and dirty class for on the fly metrics but it's now extremely deprecated."""

import os, shutil
import pandas as pd

class DataController():
    """A simple PRIVATE class for my data metrics gathering"""
    CARS = os.listdir("dataset/train")
        
    def place_cars_correctly(self):

        for car in self.CARS:
            carPics = os.listdir(f"dataset/val/{car}/")
            carPics = carPics[:10]
            os.mkdir(f"dataset/test/{car}")
            for carPic in carPics:
                shutil.move(f"dataset/val/{car}/{carPic}", f"dataset/test/{car}/{carPic}")

    def get_data_metrics(self):
        totalCarCount = 0
        lowest = 2000
        lowestCar = ''
        for car in self.CARS:
            localCount = len(os.listdir(f"dataset/train/{car}/"))
            totalCarCount = totalCarCount + localCount
            if localCount < lowest:
                lowest = localCount
                lowestCar = car

        print("Total: ", totalCarCount)
        print("Avg img count per car: ", totalCarCount/70)
        print(f"lowest count: {lowest}, for {lowestCar}")

    def create_df(self):
        dfFromWish = {
            "car": [],
            "train": [],
            "val": [],
            "test": [],
        }

        for car in self.CARS:
            dfFromWish['car'].append(car)
            dfFromWish['train'].append(len(os.listdir(f"dataset/train/{car}/")))
            dfFromWish['val'].append(len(os.listdir(f"dataset/val/{car}/")))
            dfFromWish['test'].append(len(os.listdir(f"dataset/test/{car}/")))

        df = pd.DataFrame.from_dict(dfFromWish)
        df.to_csv("dataset/dataset.csv")

if __name__ == "__main__":
    dc = DataController()
    dc.create_df()