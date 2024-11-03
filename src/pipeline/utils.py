import os
import sys

import numpy as np
import pandas as pd
import dill #Helps to create pkl file
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.pipeline.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True) 
        
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e,sys)
    
def evaluate_models(X_train, y_train,X_test,y_test,models,param):
    try:
        report = {}

        # Iterate over the models and their corresponding names
        for model_name, model in models.items():
            para=param.get(model_name,{})

            if not para:
                continue
            
            #GridSearchCV for hyperparameter tuning
            gs = GridSearchCV(model,para,cv=3)
            gs.fit(X_train,y_train)

            model.set_params(**gs.best_params_)
            #Fit the model before making predictions
            model.fit(X_train,y_train)
            
            #predictions on training and test sets
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            #calculating R2 scores for training and test data
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)
            
            #test model score in the report store
            report[model_name] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)