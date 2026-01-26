import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer  # Used to handle NAN Values with Mean,mode
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
import os
from src.exceptions import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_ob_file_path= os.path.join('artifact','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_obj(self):
        '''
        Responsible For Data Transformation
        
        '''
        try:
            numerical_columns=['reading_score', 'writing_score']
            categorical_columns=['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch','test_preparation_course']

            num_pipeline= Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy='median')),   # HANDLING THE MISSING VALUES
                    ("scaler",StandardScaler(with_mean=False))
                ]
            )

            cat_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy='most_frequent')), # HANDLING THE MISSING VALUES
                    ("one_hot_encoder",OneHotEncoder()),
                    ("Scaler",StandardScaler(with_mean=False))
                ]
            )

            logging.info("Numerical Columns encoding completed")
            logging.info("Categorical Columns encoding completed")

            logging.info(f"Numerical Columns {numerical_columns}")
            logging.info(f"Categorical Columns {categorical_columns}")


            #Transforms the column for model trainning ready
            preprocessor=ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline,numerical_columns),
                    ("cat_pipeline",cat_pipeline,categorical_columns)
                ]
            )
            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)


    def intiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            logging.info("Read Train and Test data compelted")


            logging.info("Obtaining preprocessing object")

            preprocessor_obj=self.get_data_transformer_obj()

            target_column_name="math_score"
            numerical_columns=['reading_score', 'writing_score']
            categorical_columns=['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch','test_preparation_course']

            input_features_train_df=train_df.drop(columns=[target_column_name])
            target_features_train_df=train_df[target_column_name]

            input_features_test_df=test_df.drop(columns=[target_column_name])
            target_features_test_df=test_df[target_column_name]

            logging.info("Applying Preprocessing object on traing dataframe and testing dataframe")

            input_features_train_arr=preprocessor_obj.fit_transform(input_features_train_df)
            input_features_test_arr=preprocessor_obj.transform(input_features_test_df)

            train_arr= np.c_[
                input_features_train_arr,np.array(target_features_train_df)
            ]
            test_arr= np.c_[
                input_features_test_arr,np.array(target_features_test_df)
            ]

            logging.info("Saved Preprocessing object")

            
            #SAVING THE PKL FILE
            save_object(

                file_path=self.data_transformation_config.preprocessor_ob_file_path,
                obj=preprocessor_obj,
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_ob_file_path
            )

        except Exception as e:
            raise CustomException(e,sys)       

