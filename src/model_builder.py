from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
import numpy as np
import joblib
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ModelBuilder:
    def __init__(self):
        self.model = None

    def build_model(self, X, y):
        logging.info("Building model...")
        param_dist = {
            'n_estimators': [100, 200, 300, 400, 500],
            'max_depth': [10, 20, 30, 40, 50, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'bootstrap': [True, False]
        }

        rf = RandomForestRegressor(random_state=42)
        
        self.model = RandomizedSearchCV(estimator=rf, param_distributions=param_dist, 
                                        n_iter=100, cv=3, verbose=2, random_state=42, n_jobs=-1)
        
        self.model.fit(X, y)
        logging.info(f"Best parameters: {self.model.best_params_}")
        return self.model

    def save_model(self, path):
        if self.model is not None:
            timestamp = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
            filename = f"{path}/model_{timestamp}.pkl"
            joblib.dump(self.model, filename)
            logging.info(f"Model saved to {filename}")
        else:
            logging.error("No model to save. Please build the model first.")

    def load_model(self, filename):
        self.model = joblib.load(filename)
        logging.info(f"Model loaded from {filename}")
        return self.model

    def predict(self, X):
        if self.model is not None:
            return self.model.predict(X)
        else:
            logging.error("No model available. Please build or load a model first.")
            return None