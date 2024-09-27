import os 
import sys
import pandas as pd 
import numpy as np
from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException

from src.utils import load_object


class PredictPipline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            ## Load pickel File
            ## This Code Work in /any system
            preprocessor_path = os.path.join("artifcats","preprocessor.pkl")
            model_path = os.path.join("artifcats","model.pkl")

            ## Load Pickel File
            preprocessor = load_object(preprocessor_path)
            model = load_object(model_path)

            data_scaled = preprocessor.transform(features)

            pred = model.predict(data_scaled)

            return pred

        except Exception as e:
            logging.info("Error Occure in Prediction Pipline")
            raise CustomException(e, sys)


class CustomData:
    def __init__(self,
            Gender:int,
            Age:int,
            shoulder:int,
            chest:int,
            waist:int,
            hips:int,
            shoulder_to_waist:int
            ):


        self.Gender = Gender
        self.Age= Age
        self.shoulder = shoulder
        self.chest = chest
        self.waist = waist
        self.hips = hips
        self.shoulder_to_waist = shoulder_to_waist
        


    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                'Gender':[self.Gender], 
                'Age':[self.Age],
                'ShoulderWidth':[self.shoulder],
                'ChestWidth ' : [self.chest],
                'Waist ' : [self.waist],
                'Hips ' : [self.hips],
                'ShoulderToWaist ' : [self.shoulder_to_waist]
            }

            data = pd.DataFrame(custom_data_input_dict)
            logging.info("Data Frame Gathered")
            return data

        except Exception as e:
            logging.info("Error Occured In Predict Pipline")
            raise CustomException(e, sys)

        