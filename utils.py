import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import streamlit as st
import traceback

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
    Calculate Mean Absolute Percentage Error (MAPE) without using epsilon
    
    Args:
        y_true: Array of actual values
        y_pred: Array of predicted values
        
    Returns:
        MAPE value as a percentage
    """
    # Filter out any zero values in y_true to avoid division by zero
    non_zero_indices = y_true != 0
    
    if not any(non_zero_indices):
        # If all true values are zero, return a special value
        return float('inf')
    
    # Calculate APE only for non-zero true values
    ape = np.abs((y_true[non_zero_indices] - y_pred[non_zero_indices]) / y_true[non_zero_indices]) * 100
    
    # Return the mean
    return np.mean(ape)

def train_model(df, feature_cols, random_state=123):
    """Train Random Forest model with optimized parameters"""
    """Preprocess data"""
    X, y, df_clean, imputer, scaler = preprocess_data(df, feature_cols)
   
    if X is None or y is None:
        return None, None, None, None, None
   
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
        y_pred = model.predict(X_test)
        r_squared = r2_score(y_test, y_pred)
       
        """Calculate RMSE and MAPE on test data"""
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mape = calculate_mape(y_test, y_pred)
        
        """Store imputer and scaler in model object for prediction"""
        model._imputer = imputer
        model._scaler = scaler
       
        return model, r_squared, rmse, df_clean, mape
    except Exception as e:
        st.error(f"Error during model training: {e}")
        st.error(traceback.format_exc())
        return None, None, None, None, None

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

def calculate_yearly_mape(df, model, feature_cols, target_col='btc_price_usd'):
    """
    Calculate MAPE, RMSE, and R² for each year in the dataset
    
    Args:
        df: DataFrame with date column and features
        model: Trained Random Forest model
        feature_cols: List of feature column names
        target_col: Target column name
        
    Returns:
        Dictionary with yearly statistics
    """
    if 'date' not in df.columns or model is None:
        return None
        
    try:
        # Convert date to datetime if it's not already
        if not pd.api.types.is_datetime64_any_dtype(df['date']):
            df['date'] = pd.to_datetime(df['date'])
            
        # Get unique years in the dataset
        years = df['date'].dt.year.unique()
        
        # Dictionary to store yearly metrics
        yearly_metrics = {}
        
        # Use the model's scaler stored during training
        scaler = model._scaler
        
        for year in sorted(years):
            # Filter data for this year
            year_data = df[df['date'].dt.year == year]
            
            # Skip years with insufficient data
            if len(year_data) < 5:
                continue
                
            # Prepare features and target
            X_year = year_data[feature_cols]
            y_year = year_data[target_col]
            
            # Replace 'No data' with NaN and convert to numeric
            X_year = X_year.replace('No data', np.nan)
            for col in X_year.columns:
                X_year[col] = pd.to_numeric(X_year[col], errors='coerce')
            
            # Skip if there are no valid target values
            if y_year.isna().all():
                continue
                
            # Create a new imputer for each year
            imputer = SimpleImputer(strategy='median')
            X_year_imputed = imputer.fit_transform(X_year)
            
            # Scale features using the same scaler as the model training
            X_year_scaled = scaler.transform(X_year_imputed)
            
            # Make predictions
            y_year_pred = model.predict(X_year_scaled)
            
            # Calculate metrics using scikit-learn functions
            mape = calculate_mape(y_year.values, y_year_pred)
            rmse = np.sqrt(mean_squared_error(y_year.values, y_year_pred))
            r2 = r2_score(y_year.values, y_year_pred)
            
            # Store metrics in dictionary
            yearly_metrics[year] = {
                'n': len(year_data),
                'mape': mape,
                'rmse': rmse,
                'r2': r2
            }
            
        return yearly_metrics
    except Exception as e:
        import traceback
        print(f"Error calculating yearly MAPE: {e}")
        print(traceback.format_exc())
        return None


