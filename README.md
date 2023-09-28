# FCC
Fine Car Classifier is a classification project that is trained on 70+ models of cars and would identify each ones make and model


# Model
The model architecture is extremely straightforward and simple, for the low low price of 15862006 parameters you get the following:

{insert model arch image}

The architecture is cited in this paper https://doi.org/10.48550/arXiv.2304.00501 **AKA YOLOV8-CLS**

# Data
Data was broken into train, val and test.

### Val and Test
Val set had 90 samples per car while test set had 10

### Training:
Now for the training data:

#### Initial dataset
- total training images: 53453
- average image count per car: 763
{insert init data graph}
{insert lowest cars numbers}

#### Testing for initial data
each car model was testing against 10 never seen images and these are the scores for all of them
{insert all car test}

as you can see, there are a bunch that didn't perform well, let's expand on those.
{insert lowest test scores}


### Notes:
- famous proven solution using vgg16 https://medium.com/@albionkrasniqi22_80133/vehicle-classification-742403117f43