import os
import sys
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor,RandomForestRegressor
from sklearn.metrics import r2_score

from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

from src.exceptions import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_model

from dataclasses import dataclass


@dataclass
class ModelTrainerConfig:
    trained_model_file_path= os.path.join("artifact","model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config= ModelTrainerConfig()

    def initiate_model_trainer(self, training_arr, test_arr):
        try:
            logging.info("Splitting the train and test data")

            X_train, y_train, X_test, y_test = (
                training_arr[:, :-1],
                training_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1]
            )

            models = {
                "Linear Regression": LinearRegression(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest Regressor": RandomForestRegressor(),
                "XGBRegressor": XGBRegressor(),
                "CatBoostRegressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor()
            }

            model_report = evaluate_model(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models
            )

            best_model_name = max(model_report, key=model_report.get)
            best_model_score = model_report[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found")

            best_model = models[best_model_name]

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)

            return r2_square, best_model_name

        except Exception as e:
            raise CustomException(e, sys)
