import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from datetime import datetime, timedelta
import logging
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def easter(year):
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return datetime(year, month, day)
class PreprocessingError(Exception):
    """Custom exception for preprocessing errors"""
    pass
class Preprocessor:
    def __init__(self):
   
        self.scaler = StandardScaler()
        self.num_imputer = SimpleImputer(strategy='mean')
        self.cat_imputer = SimpleImputer(strategy='constant', fill_value='missing')
        self.label_encoders = {}
        self.onehot = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        self.holidays = self._generate_holidays()
        logging.info(f"Initialized Preprocessor with {len(self.holidays)} holidays")
    def _generate_holidays(self):
        holidays = []
        for year in range(2013, 2016):
            holidays.extend([
                datetime(year, 1, 1),    # New Year's Day
                easter(year),            # Easter
                datetime(year, 10, 31),  # Halloween
                datetime(year, 12, 25)   # Christmas
            ])
        return pd.to_datetime(holidays)

    def preprocess(self, df):
        try:
            date_column = df['Date']  # Store the Date column separately
            df = self._handle_missing_values_and_encode(df)
            df['Date'] = date_column  # Add the Date column back
            df = self._extract_datetime_features(df)
        except PreprocessingError as e:
            logging.error(f"Preprocessing failed: {str(e)}")
            raise

        logging.info(f"Final DataFrame shape: {df.shape}")
        logging.info("Preprocessing completed.")
        logging.info("Data types after preprocessing:")
        # for col in df.columns:
        #     logging.info(f"{col}: {df[col].dtype}")

        return df   
    
    def _handle_missing_values_and_encode(self, df):
        start_time = time.time()
        
        logging.info("Handling missing values, encoding categorical variables, and scaling numerical variables...")
        
        try:
            # Define expected column types
            expected_numerical_columns = ['Sales', 'Customers', 'Open', 'Promo', 'SchoolHoliday', 'CompetitionDistance', 
                                          'CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear', 'Promo2', 'Promo2SinceWeek', 'Promo2SinceYear']
            expected_label_encode_columns = ['Store', 'DayOfWeek']
            expected_onehot_encode_columns = ['StateHoliday', 'StoreType', 'Assortment', 'PromoInterval']
            
            # Filter columns that actually exist in the DataFrame
            numerical_columns = [col for col in expected_numerical_columns if col in df.columns]
            label_encode_columns = [col for col in expected_label_encode_columns if col in df.columns]
            onehot_encode_columns = [col for col in expected_onehot_encode_columns if col in df.columns]
            
            # Handle numerical columns
            for col in numerical_columns:
                df[col] = self.num_imputer.fit_transform(df[[col]]).ravel()
                df[col] = self.scaler.fit_transform(df[[col]]).ravel()

            # Handle label encoding
            for col in label_encode_columns:
                df[col] = df[col].astype(str)  # Convert to string
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
                self.label_encoders[col] = le

            # Handle one-hot encoding
            df_onehot = pd.DataFrame()
            for col in onehot_encode_columns:
                df[col] = df[col].astype(str)  # Convert to string
                df[col] = self.cat_imputer.fit_transform(df[[col]]).ravel()
                onehot_cols = self.onehot.fit_transform(df[[col]])
                onehot_df = pd.DataFrame(onehot_cols, columns=[f"{col}_{cat}" for cat in self.onehot.categories_[0]])
                df_onehot = pd.concat([df_onehot, onehot_df], axis=1)

            # Combine all processed columns
            df_processed = pd.concat([df[numerical_columns + label_encode_columns], df_onehot], axis=1)
            
            logging.info(f"Missing value handling, encoding, and scaling completed in {time.time() - start_time:.2f} seconds")
            logging.info(f"DataFrame shape after processing: {df_processed.shape}")
            
            return df_processed

        except Exception as e:
            error_msg = f"Error in _handle_missing_values_and_encode: {str(e)}"
            logging.error(error_msg)
            raise PreprocessingError(error_msg)
    
    def _extract_datetime_features(self, df):
        try:
            # Check if the 'Date' column exists
            # if 'Date' not in df.columns:
            #     logging.info(f"columns.. {df.columns.tolist()}")
            #     raise KeyError("The 'Date' column is missing from the DataFrame.")
            
            df['Date'] = pd.to_datetime(df['Date'])
            df['Year'] = df['Date'].dt.year
            df['Month'] = df['Date'].dt.month
            df['Day'] = df['Date'].dt.day
            df['DayOfWeek'] = df['Date'].dt.dayofweek

            return df

        except Exception as e:
            error_msg = f"Error in _extract_datetime_features: {str(e)}"
            logging.error(error_msg)
            raise PreprocessingError(error_msg)


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