from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
import numpy as np
import joblib
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ModelBuilderError(Exception):
    """Custom exception for model building errors"""
    pass

class ModelBuilder:
    def __init__(self):
        self.model = None

    def build_model(self, X, y):
        logging.info("Building model...")
        
        # Check if 'Date' column is present in X
        if 'Date' in X.columns:
            logging.warning("'Date' column found in feature set. This column will be dropped.")
            X = X.drop('Date', axis=1)
        
        # Check data types
        non_numeric_cols = X.select_dtypes(exclude=[np.number]).columns
        if len(non_numeric_cols) > 0:
            error_msg = f"Non-numeric columns found in features: {non_numeric_cols.tolist()}"
            logging.error(error_msg)
            raise ModelBuilderError(error_msg)
        
        # Log the final set of features being used
        logging.info(f"Features used for modeling: {X.columns.tolist()}")
        
        param_dist = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2],
            'bootstrap': [True, False]
        }

        # rf = RandomForestRegressor(random_state=42)
        # Initialize the model with specified parameters
        self.model = RandomForestRegressor(n_estimators=200, max_depth = 40, random_state=42)
        
              
        try:
            self.model.fit(X, y)
            logging.info(f"Model built successfully.")
            return self.model
        except Exception as e:
            error_msg = f"Error in build_model: {str(e)}"
            logging.error(error_msg)
            raise ModelBuilderError(error_msg)

    def save_model(self, path):
        if self.model is not None:
            try:
                timestamp = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
                filename = f"{path}/model_{timestamp}.pkl"
                joblib.dump(self.model, filename)
                logging.info(f"Model saved to {filename}")
            except Exception as e:
                error_msg = f"Error in save_model: {str(e)}"
                logging.error(error_msg)
                raise ModelBuilderError(error_msg)
        else:
            logging.error("No model to save. Please build the model first.")

    def load_model(self, filename):
        try:
            self.model = joblib.load(filename)
            logging.info(f"Model loaded from {filename}")
            return self.model
        except Exception as e:
            error_msg = f"Error in load_model: {str(e)}"
            logging.error(error_msg)
            raise ModelBuilderError(error_msg)

    def predict(self, X):
        if self.model is not None:
            try:
                return self.model.predict(X)
            except Exception as e:
                error_msg = f"Error in predict: {str(e)}"
                logging.error(error_msg)
                raise ModelBuilderError(error_msg)
        else:
            logging.error("No model available. Please build or load a model first.")
            return None