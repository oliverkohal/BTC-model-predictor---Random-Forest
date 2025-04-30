import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import warnings
import traceback

"""Suppress warnings"""
warnings.filterwarnings('ignore')

"""Function to load data""" 
def load_data():
    try:
        df = pd.read_csv('btc_macroeconomic.csv')
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        return df
    except FileNotFoundError:
        st.warning("btc_macroeconomic.csv not found. Using sample data.")
        return None

"""Function to preprocess data and ensure it's suitable for training""" 
def preprocess_data(df, feature_cols, target_col='btc_price_usd'):
    """Create a copy to avoid modifying the original""" 
    df_copy = df.copy()
    
    """Ensure all feature columns and target column exist""" 
    missing_cols = [col for col in feature_cols + [target_col] if col not in df_copy.columns]
    if missing_cols:
        st.error(f"Missing columns in dataset: {', '.join(missing_cols)}")
        return None, None, None
    
    """Remove rows with NaN values in features or target""" 
    df_clean = df_copy.dropna(subset=feature_cols + [target_col])
    
    if len(df_clean) < 10:
        st.error("Not enough clean data points for reliable model training")
        return None, None, None
    
    """Ensure all data is numeric (convert to float to avoid issues with integers)""" 
    for col in feature_cols + [target_col]:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    
    """Drop any rows that couldn't be converted to numeric""" 
    df_clean = df_clean.dropna(subset=feature_cols + [target_col])
    
    """ Extract features and target""" 
    X = df_clean[feature_cols].astype(float).values  # Explicit conversion to numpy array of floats
    y = df_clean[target_col].astype(float).values    # Explicit conversion to numpy array of floats
    
    return X, y, df_clean

def train_model(df, feature_cols, random_state=123):
    """Preprocess data""" 
    X, y, df_clean = preprocess_data(df, feature_cols)
    
    if X is None or y is None:
        return None, None, None, None
    
    try:
        """Split data into training and testing sets""" 
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=random_state)
        
        """Train model""" 
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        """Calculate R-squared on test data""" 
        r_squared = model.score(X_test, y_test)
        
        """Calculate RMSE on test data""" 
        y_pred = model.predict(X_test)
        rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
        
        return model, r_squared, rmse, df_clean
    except Exception as e:
        st.error(f"Error during model training: {e}")
        return None, None, None, None

""" Function to make a prediction with multiple features""" 
def make_prediction(model, feature_values):
    if model is None:
        return None
    
    try:
        """ Ensure all features are float""" 
        features = [float(val) for val in feature_values]
        """ Reshape for sklearn""" 
        features_array = np.array(features).reshape(1, -1)
        """ Make prediction""" 
        prediction = model.predict(features_array)[0]
        return prediction
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        return None
