# FCC
Fine Car Classifier is a classification project that is trained on 70+ models of cars and would identify each ones make and model

# Usage
```python
from FCC.CarClassifier import CarClassifier

# start classifier with trained model
fcc = CarClassifier("path/to/model.pt")

# get predicted class with image path
predictedCar = fcc.predict("path/to/image.jpg")

# or get prediction with image object
ret, frame = cap.read()
predictedCar = fcc.predict(frame)
```

# Model
The model architecture is extremely straightforward and simple, for the low low price of 15862006 parameters you get the following:

![model architecure](https://github.com/karimkohel/FCC/blob/main/docs/arch.png?raw=true)

The architecture is explained in [this paper](https://doi.org/10.48550/arXiv.2304.00501) **AKA YOLOV8-CLS**, normally it's pretrained on 1000 classes from the ImageNet dataset

# Data
Data was broken into train, val and test.

### Val and Test
Val set had 90 samples per car while test set had 10

### Training:
Now for the training data:

#### Initial dataset
- total training images: 53453
- average image count per car: 763

![all data](https://github.com/karimkohel/FCC/blob/main/docs/initData.png?raw=true)
##### The lowest of which
![Lowest count cars](https://github.com/karimkohel/FCC/blob/main/docs/initDataLowest.png?raw=true)

#### Testing for initial data
each car model was testing against 10 never seen images and these are the scores for all of them
![testing scores for all cars](https://github.com/karimkohel/FCC/blob/main/docs/testingScores.png?raw=true)

as you can see, there are a bunch that didn't perform well, let's expand on those.
![lowest test scores](https://github.com/karimkohel/FCC/blob/main/docs/worstTestingScores.png?raw=true)


What does it look like when the model fails?
![bad example](https://github.com/karimkohel/FCC/blob/main/docs/badexample.jpg?raw=true)

### Notes:
- famous proven solution using vgg16 https://medium.com/@albionkrasniqi22_80133/vehicle-classification-742403117f43