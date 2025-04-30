import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import streamlit as st
import traceback
import joblib

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

def train_model(df, feature_cols, random_state=123):
    """Preprocess data"""
    X, y, df_clean, imputer, scaler = preprocess_data(df, feature_cols)
   
    if X is None or y is None:
        return None, None, None, None
   
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
       
        """Calculate RMSE on test data"""
        y_pred = model.predict(X_test)
        rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
       
        # Return only the 4 values that Streamlit expects, plus scaler separately
        # The Streamlit app will need to be modified to accept the scaler
        return model, r_squared, rmse, df_clean, scaler, imputer
    except Exception as e:
        st.error(f"Error during model training: {e}")
        st.error(traceback.format_exc())
        return None, None, None, None, None, None

"""Function to make a prediction with multiple features - ORIGINAL, KEEP FOR COMPATIBILITY"""
def make_prediction(model, feature_values):
    if model is None:
        return None
   
    try:
        """Ensure all features are float"""
        features = [float(val) for val in feature_values]
        """Reshape for sklearn"""
        features_array = np.array(features).reshape(1, -1)
        """Make prediction"""
        prediction = model.predict(features_array)[0]
        return prediction
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        st.error(traceback.format_exc())
        return None

"""New consistent prediction function"""
def predict_btc_price(model, scaler, features):
    """
    Make a prediction using the model with consistent preprocessing
    
    Args:
        model: Trained RandomForestRegressor model
        scaler: Fitted StandardScaler used during training
        features: List of input feature values
    
    Returns:
        Predicted BTC price
    """
    try:
        # Convert to correct format and apply scaling
        features_array = np.array([float(val) for val in features]).reshape(1, -1)
        features_scaled = scaler.transform(features_array)
        prediction = model.predict(features_scaled)[0]
        return prediction
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        st.error(traceback.format_exc())
        return None

"""Function to get feature importance"""
def get_feature_importance(model, feature_cols):
    if model is None or feature_cols is None:
        return None
        
    try:
        importances = model.feature_importances_
        feature_importance = dict(zip(feature_cols, importances))
        sorted_importance = {k: v for k, v in sorted(feature_importance.items(), 
                                                    key=lambda item: item[1], 
                                                    reverse=True)}
        return sorted_importance
    except Exception as e:
        st.error(f"Error getting feature importance: {e}")
        return None

"""Function to save model and preprocessing objects"""
def save_model(model, scaler, imputer, feature_cols, filename='btc_rf_model.pkl'):
    """
    Save trained model and its preprocessing components
    
    Args:
        model: Trained RandomForestRegressor
        scaler: Fitted StandardScaler
        imputer: Fitted SimpleImputer
        feature_cols: List of feature column names
        filename: Name of file to save model to
    
    Returns:
        True if successful, False otherwise
    """
    try:
        joblib.dump({
            'model': model,
            'scaler': scaler,
            'imputer': imputer,
            'feature_cols': feature_cols
        }, filename)
        return True
    except Exception as e:
        st.error(f"Error saving model: {e}")
        st.error(traceback.format_exc())
        return False

"""Function to load saved model"""
def load_model(filename='btc_rf_model.pkl'):
    """
    Load trained model and its preprocessing components
    
    Args:
        filename: Name of file to load model from
    
    Returns:
        model, scaler, imputer, feature_cols if successful, None otherwise
    """
    try:
        model_data = joblib.load(filename)
        return (
            model_data['model'],
            model_data['scaler'],
            model_data['imputer'],
            model_data['feature_cols']
        )
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.error(traceback.format_exc())
        return None, None, None, None
    
