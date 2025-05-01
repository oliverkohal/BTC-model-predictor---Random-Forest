import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import streamlit as st
import traceback
from datetime import datetime

def load_data():
    """Function to load data"""
    try:
        df = pd.read_csv('btc_macroeconomic.csv')
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        return df
    except FileNotFoundError:
        st.warning("btc_macroeconomic.csv not found. Using sample data.")
        return None

def preprocess_data(df, feature_cols, target_col='btc_price_usd'):
    """Function to preprocess data and ensure it's suitable for training"""
    """Create a copy to avoid modifying the original"""
    df_copy = df.copy()
   
    """Ensure all feature columns and target column exist"""
    missing_cols = [col for col in feature_cols + [target_col] if col not in df_copy.columns]
    if missing_cols:
        st.error(f"Missing columns in dataset: {', '.join(missing_cols)}")
        return None, None, None, None, None
   
    """Replace 'No data' with NaN and convert to numeric"""
    numeric_columns = df_copy.columns.drop('date') if 'date' in df_copy.columns else df_copy.columns
    df_copy[numeric_columns] = df_copy[numeric_columns].apply(pd.to_numeric, errors='coerce')
    df_copy = df_copy.replace('No data', np.nan)
    
    """Extract features and target before imputation"""
    X_orig = df_copy[feature_cols]
    y = df_copy[target_col]
    
    """Use median imputation for missing values"""
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X_orig)
    
    """Scale the features for better model performance"""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    
    """Remove rows with NaN values in target"""
    mask = ~np.isnan(y.values)
    X_clean = X_scaled[mask]
    y_clean = y.values[mask]
    
    """Keep the clean dataframe for UI display"""
    df_clean = df_copy.loc[mask].copy()
    
    if len(y_clean) < 10:
        st.error("Not enough clean data points for reliable model training")
        return None, None, None, None, None
    
    return X_clean, y_clean, df_clean, imputer, scaler

def calculate_mape(y_true, y_pred):
    """
    Calculate Mean Absolute Percentage Error (MAPE)
    MAPE = (100/n) * sum(|actual - predicted|/|actual|)
    
    Args:
        y_true: Array of actual values
        y_pred: Array of predicted values
        
    Returns:
        MAPE value as a percentage
    """
    # Convert inputs to numpy arrays if they aren't already
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # Avoid division by zero by excluding zero values from calculation
    mask = y_true != 0
    if not np.any(mask):
        return np.nan  # Return NaN if all actual values are zero
    
    # Calculate MAPE for non-zero values
    return 100 * np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))

def train_model(df, feature_cols, random_state=123):
    """Train Random Forest model with optimized parameters"""
    """Preprocess data"""
    X, y, df_clean, imputer, scaler = preprocess_data(df, feature_cols)
   
    if X is None or y is None:
        return None, None, None, None, None, None
   
    try:
        """Split data into training and testing sets"""
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=random_state)
        
        """Train Random Forest model with optimized parameters"""
        model = RandomForestRegressor(
            n_estimators=50,      # Using 50 trees
            max_depth=5,          # Limited depth to reduce overfitting
            min_samples_split=20, # Minimum samples required to split a node
            min_samples_leaf=3,   # Minimum samples required at a leaf node
            max_features=0.7,     # 70% of features for each split
            bootstrap=True,       # Use bootstrap samples
            oob_score=True,       # Use out-of-bag samples to estimate score
            random_state=random_state
        )
        model.fit(X_train, y_train)
       
        """Calculate R-squared on test data"""
        r_squared = model.score(X_test, y_test)
       
        """Calculate RMSE and MAPE on test data"""
        y_pred = model.predict(X_test)
        rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
        mape = calculate_mape(y_test, y_pred)
        
        """Store test predictions for later date-filtered metrics"""
        test_predictions = {
            'y_test': y_test,
            'y_pred': y_pred
        }
        
        """Store imputer and scaler in model object for prediction"""
        model._imputer = imputer
        model._scaler = scaler
       
        return model, r_squared, rmse, df_clean, mape, test_predictions
    except Exception as e:
        st.error(f"Error during model training: {e}")
        st.error(traceback.format_exc())
        return None, None, None, None, None, None

def make_prediction(model, feature_values):
    """Function to make a prediction with multiple features"""
    if model is None:
        return None
   
    try:
        """Ensure all features are float"""
        features = np.array([float(val) for val in feature_values]).reshape(1, -1)
        
        """Apply the same preprocessing as during training"""
        if hasattr(model, '_imputer') and hasattr(model, '_scaler'):
            features_imputed = model._imputer.transform(features)
            features_scaled = model._scaler.transform(features_imputed)
            prediction = model.predict(features_scaled)[0]
        else:
            # Fallback if no preprocessing objects stored
            prediction = model.predict(features)[0]
            
        return prediction
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        st.error(traceback.format_exc())
        return None

def get_feature_importance(model, feature_cols):
    """Function to get feature importance"""
    if model is None or feature_cols is None:
        return None
        
    try:
        """Get feature importances from model"""
        importances = model.feature_importances_
        
        """Create dictionary mapping feature names to importance values"""
        feature_importance = dict(zip(feature_cols, importances))
        
        """Sort by importance values in descending order"""
        sorted_importance = {k: v for k, v in sorted(feature_importance.items(), 
                                                  key=lambda item: item[1], 
                                                  reverse=True)}
        
        return sorted_importance
    except Exception as e:
        st.error(f"Error getting feature importance: {e}")
        return None

def train_filtered_model(df, feature_cols, date_start, date_end, random_state=123):
    """
    Train a model using only data from a specific date range
    
    Args:
        df: Original dataframe with date column
        feature_cols: List of feature column names
        date_start: Start date for filtering (string 'YYYY-MM-DD' or datetime)
        date_end: End date for filtering (string 'YYYY-MM-DD' or datetime)
        random_state: Random seed for reproducibility
        
    Returns:
        Same return values as train_model()
    """
    try:
        # Ensure we have the necessary data
        if df is None or 'date' not in df.columns:
            return None, None, None, None, None, None
            
        # Make a copy of the dataframe to avoid modifying the original
        df_copy = df.copy()
        
        # Make sure date column is datetime type
        if not pd.api.types.is_datetime64_any_dtype(df_copy['date']):
            df_copy['date'] = pd.to_datetime(df_copy['date'])
        
        # Convert string dates to datetime if needed
        if isinstance(date_start, str):
            date_start = pd.to_datetime(date_start)
        if isinstance(date_end, str):
            date_end = pd.to_datetime(date_end)
            
        # Filter the dataframe to only include the date range
        df_filtered = df_copy[df_copy['date'] >= date_start]
        df_filtered = df_filtered[df_filtered['date'] <= date_end]
        
        if len(df_filtered) < 10:
            st.warning(f"Not enough data points in the date range {date_start} to {date_end} for reliable model training")
            return None, None, None, None, None, None
            
        # Train model on the filtered data
        return train_model(df_filtered, feature_cols, random_state)
    except Exception as e:
        st.error(f"Error training filtered model: {e}")
        st.error(traceback.format_exc())
        return None, None, None, None, None, None
    
