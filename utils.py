import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import streamlit as st
import traceback
import joblib
import os
import requests

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
       
        return model, r_squared, rmse, df_clean
    except Exception as e:
        st.error(f"Error during model training: {e}")
        return None, None, None, None

"""Function to make a prediction with multiple features"""
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

"""NEW FUNCTIONS FOR GITHUB MODEL INTEGRATION"""

def download_model_from_github(github_username, repo_name, model_filename='btc_rf_model.pkl'):
    """
    Download the model file from GitHub if not already present locally
    
    Args:
        github_username: Your GitHub username
        repo_name: Your repository name
        model_filename: The name of the model file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Only download if file doesn't exist locally
        if not os.path.exists(model_filename):
            st.info(f"Downloading model from GitHub...")
            
            # Construct the raw GitHub URL
            url = f"https://raw.githubusercontent.com/{github_username}/{repo_name}/main/{model_filename}"
            
            # Download the file
            r = requests.get(url, allow_redirects=True)
            
            # Check if download was successful
            if r.status_code == 200:
                # Save the file
                with open(model_filename, 'wb') as f:
                    f.write(r.content)
                st.success(f"Model downloaded successfully!")
                return True
            else:
                st.error(f"Failed to download model: HTTP status {r.status_code}")
                return False
        else:
            # File already exists
            return True
            
    except Exception as e:
        st.error(f"Error downloading model: {e}")
        st.error(traceback.format_exc())
        return False

def load_model_from_file(model_filename='btc_rf_model.pkl'):
    """
    Load a trained model from a file
    
    Args:
        model_filename: Path to the model file
        
    Returns:
        model, scaler, features if successful, None otherwise
    """
    try:
        if os.path.exists(model_filename):
            # Load the model data
            model_data = joblib.load(model_filename)
            
            # Extract components
            model = model_data.get('model')
            scaler = model_data.get('scaler')
            features = model_data.get('features')
            
            return model, scaler, features
        else:
            st.error(f"Model file not found: {model_filename}")
            return None, None, None
            
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.error(traceback.format_exc())
        return None, None, None

def predict_btc_price(model, scaler, features, input_values):
    """
    Make a consistent prediction using a saved model with proper preprocessing
    
    Args:
        model: Trained RandomForestRegressor model
        scaler: Fitted StandardScaler (can be None)
        features: List of feature names (for validation)
        input_values: List of input values
        
    Returns:
        Predicted BTC price
    """
    try:
        # Validate input
        if len(input_values) != len(features):
            st.error(f"Expected {len(features)} input values, got {len(input_values)}")
            return None
            
        # Convert to numpy array
        features_array = np.array([input_values]).reshape(1, -1)
        
        # Apply preprocessing if scaler is provided
        if scaler is not None:
            features_array = scaler.transform(features_array)
            
        # Make prediction
        prediction = model.predict(features_array)[0]
        return prediction
        
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        st.error(traceback.format_exc())
        return None
