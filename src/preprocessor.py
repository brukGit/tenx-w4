import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Preprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.imputer = KNNImputer(n_neighbors=5)

    def preprocess(self, df):
        logging.info("Starting preprocessing...")
        df = self._handle_missing_values(df)
        df = self._extract_datetime_features(df)
        df = self._encode_categorical_features(df)
        df = self._scale_numerical_features(df)
        logging.info("Preprocessing completed.")
        return df

    def _handle_missing_values(self, df):
        logging.info("Handling missing values...")
        # Use KNN imputation for numerical columns
        numerical_columns = df.select_dtypes(include=[np.number]).columns
        df[numerical_columns] = self.imputer.fit_transform(df[numerical_columns])
        
        # For categorical columns, fill with mode
        categorical_columns = df.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            df[col].fillna(df[col].mode()[0], inplace=True)
        
        return df

    def _extract_datetime_features(self, df):
        logging.info("Extracting datetime features...")
        df['Date'] = pd.to_datetime(df['Date'])
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        df['Day'] = df['Date'].dt.day
        df['DayOfWeek'] = df['Date'].dt.dayofweek
        df['IsWeekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)
        
        # Extract days to/from holidays
        holidays = pd.to_datetime(['2013-12-25', '2014-12-25', '2015-12-25'])  # Example: Christmas
        df['DaysToHoliday'] = df['Date'].apply(lambda x: min((holiday - x).days for holiday in holidays if (holiday - x).days > 0))
        df['DaysFromHoliday'] = df['Date'].apply(lambda x: min((x - holiday).days for holiday in holidays if (x - holiday).days > 0))
        
        # Extract month period
        df['MonthPeriod'] = pd.cut(df['Day'], bins=[0, 10, 20, 31], labels=['Beginning', 'Middle', 'End'])
        
        return df

    def _encode_categorical_features(self, df):
        logging.info("Encoding categorical features...")
        categorical_columns = ['StoreType', 'Assortment', 'StateHoliday', 'MonthPeriod']
        return pd.get_dummies(df, columns=categorical_columns, drop_first=True)

    def _scale_numerical_features(self, df):
        logging.info("Scaling numerical features...")
        numerical_columns = df.select_dtypes(include=[np.number]).columns
        df[numerical_columns] = self.scaler.fit_transform(df[numerical_columns])
        return df

    def inverse_transform_sales(self, sales):
        return self.scaler.inverse_transform(sales.reshape(-1, 1)).flatten()