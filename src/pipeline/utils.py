import os
import sys

import numpy as np
import pandas as pd
import dill #Helps to create pkl file
from sklearn.metrics import r2_score

from src.pipeline.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True) 
        
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e,sys)
    
def evaluate_models(X_train, y_train,X_test,y_test,models):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            
            #Fit the model before making predictions
            model.fit(X_train,y_train)
            
            #predictions on training and test sets
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            #calculating R2 scores for training and test data
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)
            
            #test model score in the report store
            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)