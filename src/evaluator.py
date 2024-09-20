from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Evaluator:
    def __init__(self):
        pass

    def evaluate(self, y_true, y_pred):
        logging.info("Evaluating model performance...")
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        logging.info(f"MSE: {mse:.2f}")
        logging.info(f"RMSE: {rmse:.2f}")
        logging.info(f"MAE: {mae:.2f}")
        logging.info(f"R2 Score: {r2:.2f}")

        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }

    def plot_residuals(self, y_true, y_pred, save_path):
        logging.info("Plotting residuals...")
        residuals = y_true - y_pred
        
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=y_pred, y=residuals)
        plt.title('Residual Plot')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.axhline(y=0, color='r', linestyle='--')
        plt.savefig(f"{save_path}/residual_plot.png")
        plt.close()

    def plot_actual_vs_predicted(self, y_true, y_pred, save_path):
        logging.info("Plotting actual vs predicted values...")
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=y_true, y=y_pred)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        plt.title('Actual vs Predicted Values')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.savefig(f"{save_path}/actual_vs_predicted.png")
        plt.close()

    def plot_feature_importance(self, model, feature_names, save_path):
        logging.info("Plotting feature importance...")
        importances = model.best_estimator_.feature_importances_
        indices = np.argsort(importances)[::-1]

        plt.figure(figsize=(12, 8))
        plt.title("Feature Importances")
        plt.bar(range(len(importances)), importances[indices])
        plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
        plt.tight_layout()
        plt.savefig(f"{save_path}/feature_importance.png")
        plt.close()