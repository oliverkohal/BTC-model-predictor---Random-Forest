import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import warnings
import traceback

from utils import load_data, preprocess_data, train_model, make_prediction

def main():
    st.title('BTC Price Predictor Against Macro Conditions')
    st.write("""This model demonstrates how Bitcoin's price dynamics have evolved beyond the traditional 4-year cycle narrative in 2025. 
    It highlights the increasing influence of macroeconomic factors on BTC's valuation.""")
    st.write("""Use the sliders to explore various economic scenarios—from highly favorable to challenging conditions—and observe their significant impact on Bitcoin's predicted price movement.""")
    st.write("""Contains data from Mar-2015 to Mar-2025.""")
    
    # Load data
    btc_macro_df = load_data()
    
    if btc_macro_df is None or btc_macro_df.empty:
        st.error("Failed to load data. Please check your data source.")
        return
            
    # Define the specific macro features to use
    macro_features = [
        'gold_price_usd',
        'SP500',
        'fed_funds_rate',
        'US_inflation',
        'US_M2_money_supply_in_billions'
    ]

    # Verify which features are available in the dataset
    available_features = [feat for feat in macro_features if feat in btc_macro_df.columns]
    
    if not available_features:
        st.error("None of the required macro features are in the dataset.")
        # Show available columns
        st.write("Available columns:", ", ".join(btc_macro_df.columns.tolist()))
        return
    
    # Sidebar for model configuration#
    st.sidebar.header("Model Configuration")
    
    # Let user select features to include
    st.sidebar.subheader("Select Features to Include")
    selected_features = []
    
    for feature in available_features:
        if st.sidebar.checkbox(feature, value=True, key=f"feature_{feature}"):
            selected_features.append(feature)
    
    if not selected_features:
        st.error("Please select at least one feature for prediction.")
        return
    
    # Train model with selected features
    try:
        model, r_squared, rmse, clean_df = train_model(btc_macro_df, selected_features)
        
        if model is None or clean_df is None or clean_df.empty:
            st.error("Could not train model. Please check your data.")
            return

        # User input for prediction
        st.subheader("Make a Prediction")
        
        # Create input sliders for each feature
        feature_values = []
        
        # Make sure to use the actual column names from your dataset
        for feature in selected_features:  # Use the features that were selected for training
            min_val = float(clean_df[feature].min())
            max_val = float(clean_df[feature].max())
            current_val = float(clean_df[feature].median())
            
            feature_val = st.slider(
                f'{feature}',
                min_value=min_val,
                max_value=max_val,
                value=current_val,
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
        st.write("R² of 0.872 is quite strong, which means that the model accounts for approximately 87.2% of the fluctuations in Bitcoin prices.")
        st.write("RMSE of 7,929.99 suggest that on average, the model's predictions differ from the actual Bitcoin price by about $7,929.")
        
        coef_df = pd.DataFrame({
            'Feature': selected_features,
            'Coefficient': model.coef_
        })
        st.write("Feature Coefficients:")
        st.dataframe(coef_df)

       # Create expandable sections for each feature
        with st.container():
            st.markdown("**1. Inflation (coefficient: 541.73):**")
            st.write("For each percentage point increase in inflation, Bitcoin price is estimated to increase by $541.73, on average. This supports the narrative that Bitcoin serves as an inflation hedge - when fiat currencies lose purchasing power, Bitcoin tends to gain value.")
    
            st.markdown("**2. Fed funds rate (coefficient: -3,393.99):**")
            st.write("This large negative coefficient indicates that for each percentage point increase in the Fed funds rate, Bitcoin price is estimated to decrease by approximately $3,394. This relationship makes economic sense as higher interest rates:")
            st.markdown("""
            - Make yield-generating assets more attractive compared to non-yielding Bitcoin
            - Reduce liquidity in the financial system
            - Typically suppress risk appetite in markets
            """)
    
            st.markdown("**3. S&P 500 (coefficient: 27.39):**")
            st.write("For each point increase in the S&P 500 index, Bitcoin price tends to increase by about $27.39. This positive correlation suggests Bitcoin still behaves partially as a risk asset that rises with broader market optimism.")
    
            st.markdown("**4. Gold price (coefficient: 4.99):**")
            st.write("For each dollar increase in gold's price, Bitcoin price tends to increase by $4.99. This modest positive relationship suggests some connection between the two assets, supporting the \"digital gold\" narrative, but the effect is relatively small.")
    
            st.markdown("**5. M2 money supply (coefficient: -2.71):**")
            st.write("For each billion dollar increase in M2 money supply, Bitcoin price tends to decrease by $2.71. This slight negative relationship is counterintuitive since Bitcoin is often positioned as a hedge against monetary expansion. There might be lag effects not captured in the current model.")
                        
        # Add disclaimer
        st.info("Disclaimer: This tool is for educational purposes only. Cryptocurrency investments carry significant risk.")
    
    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.error(traceback.format_exc())

if __name__ == '__main__':
    main()


