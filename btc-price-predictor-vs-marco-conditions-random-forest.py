import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import warnings
import traceback

from utils import load_data, preprocess_data, train_model, make_prediction, get_feature_importance

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
    
    # Sidebar for model configuration
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

        # Get feature importance
        feature_importance = get_feature_importance(model, selected_features)

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
        
        if r_squared > 0.9:
            st.write(f"R² of {r_squared:.3f} is very strong, which means that the model accounts for approximately {r_squared*100:.1f}% of the fluctuations in Bitcoin prices.")
        elif r_squared > 0.8:
            st.write(f"R² of {r_squared:.3f} is quite strong, which means that the model accounts for approximately {r_squared*100:.1f}% of the fluctuations in Bitcoin prices.")

        st.write(f"RMSE of {rmse:,.2f} suggests that on average, the model's predictions differ from the actual Bitcoin price by about ${rmse:,.0f}.")
        
        # Display feature importance
        st.subheader("Feature Importance")
        
        # Create a bar chart for feature importance
        if feature_importance:
            features = list(feature_importance.keys())
            importance_values = list(feature_importance.values())
            
            # Create a DataFrame for Plotly
            importance_df = pd.DataFrame({
                'Feature': features,
                'Importance': importance_values
            })
            
            # Sort by importance
            importance_df = importance_df.sort_values('Importance', ascending=False)
            
            # Create bar chart
            fig = px.bar(importance_df, x='Feature', y='Importance', 
                         title='Random Forest Feature Importance',
                         color='Importance',
                         color_continuous_scale='Viridis')
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Create expandable sections for feature importance analysis
            with st.expander("Feature Importance Analysis", expanded=True):
                st.write("**Random Forest Feature Importance Analysis:**")
                
                # Monetary Supply
                if 'US_M2_money_supply_in_billions' in feature_importance:
                    m2_importance = feature_importance['US_M2_money_supply_in_billions']
                    st.markdown(f"**1. US M2 Money Supply (importance: {m2_importance:.2f}):**")
                    st.write(f"Money supply accounts for {m2_importance*100:.1f}% of the model's predictive power, strongly confirming the monetary expansion thesis for Bitcoin pricing.")
                
                # Inflation
                if 'US_inflation' in feature_importance:
                    inflation_importance = feature_importance['US_inflation']
                    st.markdown(f"**2. US Inflation (importance: {inflation_importance:.2f}):**")
                    st.write(f"At {inflation_importance*100:.1f}% importance, inflation serves as a significant driver, supporting Bitcoin's narrative as an inflation hedge.")
                
                # S&P 500
                if 'SP500' in feature_importance:
                    sp500_importance = feature_importance['SP500']
                    st.markdown(f"**3. S&P 500 (importance: {sp500_importance:.2f}):**")
                    st.write(f"S&P 500's {sp500_importance*100:.1f}% importance reveals correlation with traditional markets, suggesting Bitcoin isn't fully decoupled from broader market sentiment.")
                
                # Gold
                if 'gold_price_usd' in feature_importance:
                    gold_importance = feature_importance['gold_price_usd']
                    st.markdown(f"**4. Gold Price (importance: {gold_importance:.2f}):**")
                    st.write(f"The {gold_importance*100:.1f}% importance of gold prices indicates some relationship with traditional store-of-value assets, though significantly less than monetary factors.")
                
                # Fed Funds Rate
                if 'fed_funds_rate' in feature_importance:
                    rate_importance = feature_importance['fed_funds_rate']
                    st.markdown(f"**5. Fed Funds Rate (importance: {rate_importance:.2f}):**")
                    st.write(f"Fed Funds Rate at {rate_importance*100:.1f}% suggests interest rates have less direct impact compared to money supply and inflation.")
                
                st.write("The model strongly supports the monetary theory of Bitcoin pricing, where expanded money supply flows into assets over time, with inflation expectations acting as a secondary driver of investor behavior.")
        
        # Add disclaimer
        st.info("Disclaimer: This tool is for educational purposes only. Cryptocurrency investments carry significant risk.")
    
    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.error(traceback.format_exc())

if __name__ == '__main__':
    main()
