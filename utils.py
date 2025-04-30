import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import warnings
import traceback

"""Function to load data"""
def load_data():
    try:
        df = pd.read_csv('btc_macroeconomic.csv')
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        return df
    except FileNotFoundError:
        print("btc_macroeconomic.csv not found. Using sample data.")
        return None

"""Function to preprocess data and ensure it's suitable for training"""
def preprocess_data(df, feature_cols, target_col='btc_price_usd'):
    """Create a copy to avoid modifying the original"""
    df_copy = df.copy()
   
    """Ensure all feature columns and target column exist"""
    missing_cols = [col for col in feature_cols + [target_col] if col not in df_copy.columns]
    if missing_cols:
        print(f"Missing columns in dataset: {', '.join(missing_cols)}")
        return None, None, None, None, None
   
    """Replace 'No data' with NaN and convert to numeric"""
    numeric_columns = df_copy.columns.drop('date') if 'date' in df_copy.columns else df_copy.columns
    df_copy[numeric_columns] = df_copy[numeric_columns].apply(pd.to_numeric, errors='coerce')
    df_copy = df_copy.replace('No data', np.nan)
    
    """Extract features and target before imputation (for later reference)"""
    X = df_copy[feature_cols]
    y = df_copy[target_col]
    
    """Use median imputation for missing values"""
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X)
    
    """Scale the features for better model performance"""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    
    """Remove rows with NaN values in target"""
    mask = ~np.isnan(y.values)
    X_clean = X_scaled[mask]
    y_clean = y.values[mask]
    
    if len(y_clean) < 10:
        print("Not enough clean data points for reliable model training")
        return None, None, None, None, None
    
    return X_clean, y_clean, list(X.columns), imputer, scaler

def train_model(df, feature_cols, random_state=123):
    """Preprocess data"""
    X, y, feature_names, imputer, scaler = preprocess_data(df, feature_cols)
   
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
       
        """Calculate RMSE on test data"""
        y_pred = model.predict(X_test)
        rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
        
        """Calculate OOB score"""
        oob_score = model.oob_score_
       
        return model, r_squared, rmse, oob_score, imputer, scaler
    except Exception as e:
        print(f"Error during model training: {e}")
        traceback.print_exc()
        return None, None, None, None, None, None

"""Function to make a prediction with multiple features"""
def make_prediction(model, feature_values, imputer, scaler):
    if model is None or imputer is None or scaler is None:
        return None
   
    try:
        """Ensure all features are float and reshape for sklearn"""
        features = np.array([float(val) for val in feature_values]).reshape(1, -1)
        
        """Apply the same preprocessing as during training"""
        features_imputed = imputer.transform(features)
        features_scaled = scaler.transform(features_imputed)
        
        """Make prediction"""
        prediction = model.predict(features_scaled)[0]
        return prediction
    except Exception as e:
        print(f"Error making prediction: {e}")
        traceback.print_exc()
        return None

"""Function to display feature importance"""
def get_feature_importance(model, feature_names):
    if model is None or feature_names is None:
        return None
        
    try:
        importances = model.feature_importances_
        feature_importance = dict(zip(feature_names, importances))
        sorted_importance = {k: v for k, v in sorted(feature_importance.items(), 
                                                    key=lambda item: item[1], 
                                                    reverse=True)}
        return sorted_importance
    except Exception as e:
        print(f"Error getting feature importance: {e}")
        return None

# Example usage
if __name__ == "__main__":
    # Default feature columns
    feature_cols = ['gold_price_usd', 'SP500', 'fed_funds_rate', 
                   'US_inflation', 'US_M2_money_supply_in_billions']
    
    # Load data
    df = load_data()
    
    if df is not None:
        # Train model
        model, r_squared, rmse, oob_score, imputer, scaler = train_model(df, feature_cols)
        
        if model is not None:
            print(f"\nRandom Forest Performance:")
            print(f"Test R² Score: {r_squared:.4f}")
            print(f"Test RMSE: {rmse:.4f}")
            print(f"OOB R² Score: {oob_score:.4f}")
            
            # Display feature importance
            importance = get_feature_importance(model, feature_cols)
            if importance:
                print("\nFeature Importance:")
                for feature, score in importance.items():
                    print(f"{feature}: {score:.4f}")
            
            # Example prediction (replace with actual values)
            sample_features = [1800, 4000, 0.5, 2.5, 21500]  # Example values
            prediction = make_prediction(model, sample_features, imputer, scaler)
            if prediction is not None:
                print(f"\nSample prediction: ${prediction:.2f}")
    
