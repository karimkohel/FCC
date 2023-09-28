"""The Fine Car Classifier project is hosted and documented on github.com/karimkohe/fcc
"""
from ultralytics import YOLO
from ultralytics.engine.results import Results

class CarClassifier():
    """CarClassifier is a wrapper class for the custom trained classification model to facilitate easier usage and code integration"""

    def __init__(self, model: str) -> None:
        """Load model

        INPUT:
        ---
        model: required, string path to the model the class should use for classification

        OUTPUT:
        ---
        CarClassifier class instance 
        """

        self.model = YOLO(model)
        self.NAMES = self.model.names

    def predict(self, imgSrc: str) -> str: # TODO: change the typehinting to str or ndarray
        """Use the class classifier model to make a prediction on one image and return the predicted car model name
        
        INPUT
        ---
        imgSrc: required, the string path to the image or image object to predict it's class

        OUTPUT
        ---
        string containing the predicted class name
        """
        results = self.model.predict(source=imgSrc)
        return self.NAMES[results[0].probs.top1]