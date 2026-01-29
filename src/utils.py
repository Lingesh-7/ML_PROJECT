import os
import sys 
import numpy as np
import pandas as pd
from src.exceptions import CustomException
import dill
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV


def save_object(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)

        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,"wb") as f:
            dill.dump(obj,f)

    except Exception as e:
        raise CustomException(e,sys)
    



    
def evaluate_model(X_train, y_train, X_test, y_test, models, params):
    try:
        report = {}

        for model_name, model in models.items():

            param_grid = params.get(model_name)
            if param_grid is None:
                continue

            gs = GridSearchCV(
                estimator=model,
                param_grid=param_grid,
                cv=3,
                
            )

            gs.fit(X_train, y_train)
            print(gs.best_params_)

            best_model = gs.best_estimator_

            y_test_pred = best_model.predict(X_test)
            test_model_score = r2_score(y_test, y_test_pred)

            report[model_name] = test_model_score
            models[model_name] = best_model

        return report

    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    try:
        with open(file_path,"rb") as file_obj:
            return dill.load(file_obj)
        
    except Exception as e:
        raise CustomException(e,sys)