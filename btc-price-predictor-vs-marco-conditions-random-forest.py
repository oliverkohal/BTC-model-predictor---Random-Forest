import streamlit as st
import pandas as pd
import numpy as np
import warnings
import traceback
import os

from utils import load_data, preprocess_data, train_model, make_prediction, get_feature_importance
"""
Define feature display names mapping
"""
feature_display_names = {
    'gold_price_usd': 'Gold Price in USD',
    'SP500': 'S&P 500',
    'fed_funds_rate': 'Fed Funds Rate in %',
    'US_inflation': 'US Inflation Rate in %',
    'US_M2_money_supply_in_billions': 'US M2 Money Supply in Billions'
}

def clean_numeric_data(df, columns):
    """
    Clean numeric data by removing 'No data' and converting to float
    """
    df_clean = df.copy()
    
    # Replace 'No data' with NaN
    df_clean = df_clean.replace('No data', np.nan)
    
    # Convert to numeric
    for col in columns:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    
    return df_clean

def get_min_max_values(df, feature):
    """
    Get min and max values for a feature safely
    """
    try:
        # Handle case where all values are NaN
        if df[feature].isna().all():
            return 0, 100  # Default fallback values
            
        # Get min and max, ignoring NaN values
        min_val = float(df[feature].dropna().min())
        max_val = float(df[feature].dropna().max())
        
        # Ensure min != max to avoid slider errors
        if min_val == max_val:
            min_val = max(0, min_val - 1)
            max_val = max_val + 1
            
        return min_val, max_val
    except:
        # Fallback default values
        default_ranges = {
            'gold_price_usd': (1000, 3500),
            'SP500': (1800, 6200),
            'fed_funds_rate': (0, 6),
            'US_inflation': (-1, 9),
            'US_M2_money_supply_in_billions': (11000, 22000)
        }
        
        if feature in default_ranges:
            return default_ranges[feature]
        else:
            return 0, 100  # Generic fallback

def main():
    st.title('BTC Price Predictor Against Macro Conditions')
    st.write("""This model demonstrates how Bitcoin's price dynamics have evolved beyond the traditional 4-year cycle narrative in 2025. 
    It highlights the increasing influence of macroeconomic factors on BTC's valuation.""")
    st.write("""This also gives our users a powerful tool to simulate institutional BTC valuation models.""")
    st.write("""Use the sliders to explore various economic scenarios—from highly favorable to challenging conditions—and observe their significant impact on Bitcoin's predicted price movement.""")
    st.write("""Contains data from Mar-2015 to Mar-2025.""")
    
    # Load data
    btc_macro_df = load_data()
    
    if btc_macro_df is None or btc_macro_df.empty:
        st.error("Failed to load data. Please check your data source.")
        return
        
    # Clean the dataframe to handle 'No data' and convert to numeric
    clean_btc_df = clean_numeric_data(btc_macro_df, btc_macro_df.columns)
        
    # Define the specific macro features to use
    macro_features = [
        'gold_price_usd',
        'SP500',
        'fed_funds_rate',
        'US_inflation',
        'US_M2_money_supply_in_billions'
    ]

    # Verify which features are available in the dataset
    available_features = [feat for feat in macro_features if feat in clean_btc_df.columns]
    
    if not available_features:
        st.error("None of the required macro features are in the dataset.")
        # Show available columns
        st.write("Available columns:", ", ".join(clean_btc_df.columns.tolist()))
        return
    
    # Sidebar for model configuration
    st.sidebar.header("Model Configuration")
    
    # Let user select features to include
    st.sidebar.subheader("Select Features to Include")
    selected_features = []

    for feature in available_features:
        display_name = FEATURE_DISPLAY_NAMES.get(feature, feature)
        if st.sidebar.checkbox(display_name, value=True, key=f"feature_{feature}"):
            selected_features.append(feature)
    
    if not selected_features:
        st.error("Please select at least one feature for prediction.")
        return
    
    # Train model with selected features
    try:
        with st.spinner("Training model..."):
            model, r_squared, rmse, clean_df = train_model(clean_btc_df, selected_features)
            
            if model is None:
                st.error("Could not train model. Please check your data.")
                return
        
        # User input for prediction
        st.subheader("Make a Prediction")
        
        # Create input sliders for each feature
        feature_values = []
        
        # Make sure to use the actual column names from your dataset
        for feature in selected_features:
            min_val, max_val = get_min_max_values(clean_df, feature)
            
            # Use median as default value
            default_val = float(clean_df[feature].dropna().median())
                
            # Ensure default is within bounds
            default_val = max(min_val, min(default_val, max_val))
            
            # Use friendly display name for the slider
            display_name = FEATURE_DISPLAY_NAMES.get(feature, feature)
            
            feature_val = st.slider(
                display_name,
                min_value=min_val,
                max_value=max_val,
                value=default_val,
                step=(max_val - min_val) / 100,
                key=f"slider_{feature}"
            )
            feature_values.append(feature_val)

        # Predict button
        if st.button("Predict BTC Price"):
            prediction = make_prediction(model, feature_values)
            
            if prediction is not None:
                st.success(f'Estimated BTC price: ${prediction:,.2f}')
                
                  
        # Display model info
        st.subheader("Model Information")
        st.write(f"Model R-squared: {r_squared:.4f}")
        st.write(f"RMSE (Root Mean Square Error): ${rmse:,.2f}")
        
        if r_squared > 0.9:
            st.write(f"R² of {r_squared:.3f} is very strong, which means that the model accounts for approximately {int(r_squared*100)}% of the fluctuations in Bitcoin prices.")
        elif r_squared > 0.8:
            st.write(f"R² of {r_squared:.3f} is quite strong, which means that the model accounts for approximately {int(r_squared*100)}% of the fluctuations in Bitcoin prices.")

        st.write(f"RMSE of {rmse:,.2f} suggests that on average, the model's predictions differ from the actual Bitcoin price by about ${rmse:,.0f}.")
        
        # Display feature importance
        st.subheader("Feature Importance Analysis")  # Changed from "Feature Importance" to be more specific
        
        # Get feature importance
        feature_importance = get_feature_importance(model, selected_features)
        
        # Create a bar chart for feature importance
        if feature_importance:
            # Convert feature importance to use display names
            display_importance = {}
            for feature, importance in feature_importance.items():
                display_name = FEATURE_DISPLAY_NAMES.get(feature, feature)
                display_importance[display_name] = importance
            
            features = list(display_importance.keys())
            importance_values = list(display_importance.values())
            
            # Create a DataFrame for Plotly
            importance_df = pd.DataFrame({
                'Feature': features,
                'Importance': importance_values
            })
            
            # Sort by importance
            importance_df = importance_df.sort_values('Importance', ascending=False)
                        
            # Create mapping of display names back to original names for the conditional statements
            reverse_mapping = {v: k for k, v in FEATURE_DISPLAY_NAMES.items()}
            
            # Display each feature's importance with analysis
            for i, (display_feature, importance) in enumerate(sorted(display_importance.items(), 
                                                           key=lambda x: x[1], 
                                                           reverse=True), 1):
                # Format importance as whole percentage (no decimal)
                importance_pct = int(importance * 100)
                st.markdown(f"**{i}. {display_feature} (importance: {importance_pct}%):**")
                
                # Get original feature name for conditional logic
                original_feature = reverse_mapping.get(display_feature, display_feature)
                
                if original_feature == 'US_M2_money_supply_in_billions':
                    st.write(f"Money supply accounts for {importance_pct}% of the model's predictive power, strongly confirming the monetary expansion thesis for Bitcoin pricing.")
                
                elif original_feature == 'US_inflation':
                    st.write(f"At {importance_pct}% importance, inflation serves as a significant driver, supporting Bitcoin's narrative as an inflation hedge.")
                
                elif original_feature == 'SP500':
                    st.write(f"S&P 500's {importance_pct}% importance reveals correlation with traditional markets, suggesting Bitcoin isn't fully decoupled from broader market sentiment.")
                
                elif original_feature == 'gold_price_usd':
                    st.write(f"The {importance_pct}% importance of gold prices indicates some relationship with traditional store-of-value assets, though significantly less than monetary factors.")
                
                elif original_feature == 'fed_funds_rate':
                    st.write(f"Fed Funds Rate at {importance_pct}% suggests interest rates have less direct impact compared to money supply and inflation.")
                
                else:
                    st.write(f"This feature contributes {importance_pct}% to the model's predictive power.")
            
            st.write("The model strongly supports the monetary theory of Bitcoin pricing, where expanded money supply flows into assets over time, with inflation expectations acting as a secondary driver of investor behavior.")
        
        # Add disclaimer
        st.info("Disclaimer: This tool is for educational purposes only. Cryptocurrency investments carry significant risk.")
    
    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.error(traceback.format_exc())

if __name__ == '__main__':
    main()
