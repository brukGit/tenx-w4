import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import logging
from preprocessor import Preprocessor
from model_builder import ModelBuilder
from evaluator import Evaluator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(train_path, test_path, store_path):
    logging.info("Loading data...")
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    store = pd.read_csv(store_path)
    return train, test, store

def merge_data(train, test, store):
    logging.info("Merging data...")
    train = pd.merge(train, store, on='Store', how='left')
    test = pd.merge(test, store, on='Store', how='left')
    return train, test

def main():
    # Load and merge data
    train, test, store = load_data('../resources/Data/train.csv', 
                                   '../resources/Data/test.csv', 
                                   '../resources/Data/store.csv')
    train, test = merge_data(train, test, store)

    # Preprocess data
    preprocessor = Preprocessor()
    train_processed = preprocessor.preprocess(train)
    test_processed = preprocessor.preprocess(test)

    # Split data
    X = train_processed.drop(['Sales', 'Customers'], axis=1)
    y = train_processed['Sales']
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Build model
    model_builder = ModelBuilder()
    model = model_builder.build_model(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_val)

    # Evaluate model
    evaluator = Evaluator()
    metrics = evaluator.evaluate(y_val, y_pred)
    
    print("Model Performance Metrics:")
    for metric, value in metrics.items():
        print(f"{metric.upper()}: {value:.4f}")
    # Plot results
    evaluator.plot_residuals(y_val, y_pred, '../models/figures')
    evaluator.plot_actual_vs_predicted(y_val, y_pred, '../models/figures')
    evaluator.plot_feature_importance(model, X.columns, '../models/figures')

    # Save model
    model_builder.save_model('../models')

    # Make predictions on test set
    test_predictions = model.predict(test_processed.drop(['Sales', 'Customers'], axis=1))
    
    # Inverse transform the scaled predictions
    test_predictions = preprocessor.inverse_transform_sales(test_predictions)
    
    # Save predictions
    test['Predicted_Sales'] = test_predictions
    test[['Id', 'Predicted_Sales']].to_csv('../models/predictions/test_predictions.csv', index=False)

if __name__ == "__main__":
    main()