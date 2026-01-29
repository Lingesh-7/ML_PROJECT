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
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest": RandomForestRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "AdaBoost": AdaBoostRegressor(),
                "K-Neighbors": KNeighborsRegressor(),
                "XGBRegressor": XGBRegressor(),
                # "CatBoostRegressor": CatBoostRegressor(verbose=False)
            }


            params = {

                "Linear Regression": {
                    "fit_intercept": [True, False],
                    "positive": [True, False]
                },

                "Decision Tree": {
                    "criterion": ["squared_error", "friedman_mse"],
                    "max_depth": [None, 5, 10, 20, 30],
                    # "min_samples_split": [2, 5, 10],
                    # "min_samples_leaf": [1, 2, 4]
                },

                "Random Forest": {
                    "n_estimators": [100, 200, 300],
                    "max_depth": [None, 10, 20, 30],
                    # "min_samples_split": [2, 5, 10],
                    # "min_samples_leaf": [1, 2, 4],
                    # "max_features": ["sqrt", "log2"]
                },

                "Gradient Boosting": {
                    "n_estimators": [100, 200],
                    "learning_rate": [0.01, 0.05, 0.1],
                    # "max_depth": [3, 5, 7],
                    # "subsample": [0.8, 1.0]
                },

                "AdaBoost": {
                    "n_estimators": [50, 100, 200],
                    "learning_rate": [0.01, 0.05, 0.1, 1.0]
                },

                "K-Neighbors": {
                    "n_neighbors": [3, 5, 7, 9, 11],
                    "weights": ["uniform", "distance"],
                    "algorithm": ["auto", "ball_tree", "kd_tree"]
                },

                "XGBRegressor": {
                    "n_estimators": [100, 200],
                    "learning_rate": [0.01, 0.05, 0.1],
                    # "max_depth": [3, 5, 7],
                    # "subsample": [0.8, 1.0],
                    # "colsample_bytree": [0.8, 1.0]
                },

                # "CatBoostRegressor": {
                #     "iterations": [300, 500],
                #     "learning_rate": [0.03, 0.1],
                #     # "depth": [6, 8, 10],
                #     # "l2_leaf_reg": [1, 3, 5]
                # }
            }


            model_report = evaluate_model(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
                params=params
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
