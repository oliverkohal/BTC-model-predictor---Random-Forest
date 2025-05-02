import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
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
                
            # Apply the same preprocessing as during training
            if hasattr(model, '_imputer') and hasattr(model, '_scaler'):
                X_year_imputed = model._imputer.transform(X_year)
                X_year_scaled = model._scaler.transform(X_year_imputed)
            else:
                # Fallback if no preprocessing objects stored
                imputer = SimpleImputer(strategy='median')
                X_year_imputed = imputer.fit_transform(X_year)
                scaler = StandardScaler()
                X_year_scaled = scaler.fit_transform(X_year_imputed)
            
            # Make predictions
            y_year_pred = model.predict(X_year_scaled)
            
            # Filter out NaN values in actual values
            mask = ~np.isnan(y_year.values)
            y_actual = y_year.values[mask]
            y_pred = y_year_pred[mask]
            
            # Skip if no valid pairs remain
            if len(y_actual) < 5:
                continue
            
            # Calculate metrics
            mape = calculate_mape(y_actual, y_pred)
            rmse = np.sqrt(np.mean((y_actual - y_pred) ** 2))
            r2 = np.corrcoef(y_actual, y_pred)[0, 1] ** 2 if len(y_actual) > 1 else np.nan
            
            # Store metrics in dictionary
            yearly_metrics[year] = {
                'n': len(y_actual),
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
        r_squared = model.score(X_test, y_test)
       
        """Calculate RMSE and MAPE on test data"""
        y_pred = model.predict(X_test)
        rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
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


