import streamlit as st
import pandas as pd
import numpy as np
import traceback

from utils import load_data, train_model, make_prediction, get_feature_importance, calculate_yearly_mape

FEATURE_DISPLAY_NAMES = {
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
    df_clean = df_clean.replace('No data', np.nan)
    
    for col in columns:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    
    return df_clean

def get_min_max_values(df, feature):
    """
    Get min and max values for a feature safely
    """
    try:
        if df[feature].isna().all():
            return 0, 100
            
        min_val = float(df[feature].dropna().min())
        max_val = float(df[feature].dropna().max())
        
        if min_val == max_val:
            min_val = max(0, min_val - 1)
            max_val = max_val + 1
            
        return min_val, max_val
    except:
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
            return 0, 100

def get_safe_median(df, feature):
    """
    Get median value for a feature safely
    """
    try:
        if df[feature].isna().all():
            return 50
            
        # Get the median and ensure it's a scalar value
        median_val = df[feature].dropna().median()
        # Convert to float to ensure we're returning a scalar
        return float(median_val)
    except:
        default_medians = {
            'gold_price_usd': 2000,
            'SP500': 4000,
            'fed_funds_rate': 2,
            'US_inflation': 2,
            'US_M2_money_supply_in_billions': 16000
        }
        
        if feature in default_medians:
            return default_medians[feature]
        else:
            return 50

def sort_by_importance(items):
    """
    Sort items by importance value (second item in tuple)
    """
    def get_importance(item):
        return item[1]
    
    return sorted(items, key=get_importance, reverse=True)

# You can keep the function definition but we won't use it
def format_yearly_metrics(yearly_metrics):
    """
    Format yearly metrics for display
    """
    if not yearly_metrics:
        return "No yearly metrics available"
        
    result = "Yearly MAPE Analysis (2015-2025):\n"
    
    for year, metrics in sorted(yearly_metrics.items()):
        n = metrics['n']
        mape = metrics['mape']
        rmse = metrics['rmse']
        r2 = metrics['r2']
        
        result += f"Year {year} (n={n}): MAPE = {mape:.2f}%, RMSE = {rmse:.2f}, R² = {r2:.2f}\n"
        
    return result

def main():
    st.title('BTC Price Predictor Against Macro Conditions')
    st.write("""This model demonstrates how Bitcoin's price dynamics have evolved beyond the traditional 4-year cycle narrative in 2025. 
    It highlights the increasing influence of macroeconomic factors on BTC's valuation.""")
    st.write("""This also gives users a powerful tool to simulate institutional BTC valuation models.""")
    st.write("""Use the sliders to explore various economic scenarios—from highly favorable to challenging conditions—and observe their significant impact on Bitcoin's predicted price movement.""")
    st.write("""Contains data from Mar-2015 to Mar-2025.""")
    
    btc_macro_df = load_data()
    
    if btc_macro_df is None or btc_macro_df.empty:
        st.error("Failed to load data. Please check your data source.")
        return
        
    clean_btc_df = clean_numeric_data(btc_macro_df, btc_macro_df.columns)
        
    macro_features = [
        'gold_price_usd',
        'SP500',
        'fed_funds_rate',
        'US_inflation',
        'US_M2_money_supply_in_billions'
    ]

    available_features = [feat for feat in macro_features if feat in clean_btc_df.columns]
    
    if not available_features:
        st.error("None of the required macro features are in the dataset.")
        st.write("Available columns:", ", ".join(clean_btc_df.columns.tolist()))
        return
    
    st.sidebar.header("Model Configuration")
    
    st.sidebar.subheader("Select Features to Include")
    selected_features = []

    for feature in available_features:
        display_name = FEATURE_DISPLAY_NAMES.get(feature, feature)
        if st.sidebar.checkbox(display_name, value=True, key=f"feature_{feature}"):
            selected_features.append(feature)
    
    if not selected_features:
        st.error("Please select at least one feature for prediction.")
        return
    
    try:
        with st.spinner("Training model..."):
            model, r_squared, rmse, clean_df, mape = train_model(clean_btc_df, selected_features)
            
            if model is None:
                st.error("Could not train model. Please check your data.")
                return
                
            # Calculate yearly metrics
            yearly_metrics = calculate_yearly_mape(clean_btc_df, model, selected_features)
        
        st.subheader("Make a Prediction")
        
        feature_values = []
        
        for feature in selected_features:
            min_val, max_val = get_min_max_values(clean_df, feature)
            
            # Use median as default value - safely
            default_val = get_safe_median(clean_df, feature)
            
            # Ensure default is within bounds
            default_val = max(min_val, min(default_val, max_val))
            
            # Determine if this feature should show decimals
            if feature in ['fed_funds_rate', 'US_inflation']:
                # For Fed Funds Rate and US Inflation, keep decimals
                # Round to 2 decimal places for cleaner display
                min_val = round(min_val, 2)
                max_val = round(max_val, 2)
                default_val = round(default_val, 2)
                step = 0.01  # Use small step size for precise control
            else:
                # For other features, round to integers
                min_val = int(min_val)
                max_val = int(max_val)
                default_val = int(default_val)
                step = 1  # Use step of 1 for integer values
            
            # Use a display name for the slider
            display_name = FEATURE_DISPLAY_NAMES.get(feature, feature)
            
            feature_val = st.slider(
                display_name,
                min_value=min_val,
                max_value=max_val,
                value=default_val,
                step=step,
                key=f"slider_{feature}"
            )
            feature_values.append(feature_val)

        if st.button("Predict BTC Price"):
            prediction = make_prediction(model, feature_values)
            
            if prediction is not None:
                st.success(f'Estimated BTC price: ${prediction:,.0f}')
                
        st.subheader("Model Information")
        st.write(f"Model R-squared: {r_squared:.2f}")
        st.write(f"RMSE (Root Mean Square Error): ${rmse:,.0f}")
        if mape is not None:
            st.write(f"MAPE (Mean Absolute Percentage Error): {mape:.0f}%")
            st.write(f"MAPE means that, on average, the model's predictions are off by {mape:.0f}% of the actual Bitcoin price.")
        st.write(f"R² of {r_squared:.2f} is very strong, which means that the model accounts for approximately {int(r_squared*100)}% of the fluctuations in Bitcoin prices.")
        st.write(f"RMSE of {rmse:,.0f} suggests that on average, the model's predictions differ from the actual Bitcoin price by about ${rmse:,.0f}.")
        
        # Add yearly metrics display
        if yearly_metrics:
            st.subheader("Year-by-Year Analysis")
    
            # Create a DataFrame for better display
            yearly_df = pd.DataFrame.from_dict(yearly_metrics, orient='index')
            yearly_df.index.name = 'Year'
            yearly_df.reset_index(inplace=True)
    
            # Format the columns with exactly 2 decimal places including trailing zeros
            yearly_df['MAPE (%)'] = [f"{value:.2f}" for value in yearly_df['mape']]
            yearly_df['RMSE'] = [f"{value:.2f}" for value in yearly_df['rmse']]
            yearly_df['R²'] = [f"{value:.2f}" for value in yearly_df['r2']]
            yearly_df['Sample Size'] = yearly_df['n']
    
            # Display only relevant columns
            display_df = yearly_df[['Year', 'Sample Size', 'MAPE (%)', 'RMSE', 'R²']]
            st.table(display_df)
            
            # Add explanation of yearly metrics
            st.markdown("""
            #### Insights from Yearly Analysis
            
            - **Early Market (2015)**:  High MAPE (40.41%) and negative R² (-1.89) indicate that in Bitcoin's early days, macroeconomic factors were poor predictors of price movements
            - **Middle Period (2016-2022)**: Mixed performance with R² values ranging from 0.32 to 0.87, suggesting varying degrees of correlation
            - **Recent Years (2023-2025)**: Improving MAPE values (dropping to just 4.11% in 2025) suggest increasing alignment with macroeconomic indicators
            - **Crisis Periods**: Note the strong model performance during major economic events (e.g., 2020 pandemic). The Fed's aggressive monetary policy response (rate cuts and expanded money supply) created strong directional signals that the model could capture.
            
            This year-by-year analysis reveals Bitcoin's evolution from a speculative asset disconnected from traditional economics to one that increasingly responds to macroeconomic conditions. This is due to institutions playing a bigger role in BTC market in the later years and viewing it as a safe heaven from US dollar declines.
            """)
            
            # Removed the "Show Raw Yearly Metrics" expander section
        
        st.subheader("Feature Importance Analysis")
        
        feature_importance = get_feature_importance(model, selected_features)
        
        if feature_importance:
            display_importance = {}
            for feature, importance in feature_importance.items():
                display_name = FEATURE_DISPLAY_NAMES.get(feature, feature)
                display_importance[display_name] = importance
            
            feature_explanations = {
                'US_M2_money_supply_in_billions': "Money supply accounts for {pct}% of the model's predictive power, strongly confirming the monetary expansion thesis for Bitcoin pricing.",
                'US_inflation': "At {pct}% importance, inflation serves as a significant driver, supporting Bitcoin's narrative as an inflation hedge.",
                'SP500': "S&P 500's {pct}% importance reveals correlation with traditional markets, suggesting Bitcoin isn't fully decoupled from broader market sentiment.",
                'gold_price_usd': "The {pct}% importance of gold prices indicates some relationship with traditional store-of-value assets, though significantly less than monetary factors.",
                'fed_funds_rate': "Fed Funds Rate at {pct}% suggests interest rates have less direct impact compared to money supply and inflation."
            }
            
            reverse_mapping = {v: k for k, v in FEATURE_DISPLAY_NAMES.items()}
            
            sorted_features = sort_by_importance(display_importance.items())
            
            for i, (display_feature, importance) in enumerate(sorted_features, 1):
                importance_pct = int(importance * 100)
                st.markdown(f"**{i}. {display_feature} (importance: {importance_pct}%):**")
                
                original_feature = reverse_mapping.get(display_feature, display_feature)
                
                if original_feature in feature_explanations:
                    st.write(feature_explanations[original_feature].format(pct=importance_pct))
                else:
                    st.write(f"This feature contributes {importance_pct}% to the model's predictive power.")
            
            st.write("The model strongly supports the monetary theory of Bitcoin pricing, where expanded money supply flows into assets over time, with inflation expectations acting as a secondary driver of investor behavior.")
        
        st.info("Disclaimer: This tool is for educational purposes only. Cryptocurrency investments carry significant risk.")
    
    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.error(traceback.format_exc())

if __name__ == '__main__':
    main()
