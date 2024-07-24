import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

# Function to load and preprocess data
def load_and_preprocess_data(filepath):
    data = pd.read_csv(filepath)
    data['Datetime'] = pd.to_datetime(data['Date'] + ' ' + data['Time'])
    data = data.drop(columns=['Date', 'Time'])
    data = data.dropna()
    return data

# Function to scale the features
def scale_features(X, scaler):
    X_scaled = scaler.transform(X)
    return X_scaled

# Function to predict using the trained model
def predict(model, scaler, current_data):
    # Scale the current data using the fitted scaler
    current_data_scaled = scale_features(current_data, scaler)
    
    # Predict the health score
    predicted_score = model.predict(current_data_scaled)[0]
    
    return predicted_score

# Function to plot distributions
def plot_distributions(data):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Temperature Distribution
    sns.histplot(data['temperature'], kde=True, bins=30, color='blue', ax=axes[0])
    axes[0].set_title('Temperature Distribution')
    axes[0].set_xlabel('Temperature')
    
    # ECG Distribution
    sns.histplot(data['ecg'], kde=True, bins=30, color='green', ax=axes[1])
    axes[1].set_title('ECG Distribution')
    axes[1].set_xlabel('ECG')
    
    # Pulse Distribution
    sns.histplot(data['pulse'], kde=True, bins=30, color='red', ax=axes[2])
    axes[2].set_title('Pulse Distribution')
    axes[2].set_xlabel('Pulse')

    plt.tight_layout()
    st.pyplot(fig)

# Main function to run the Streamlit app
def main():
    # Set page title and favicon
    st.set_page_config(page_title='Health Prediction Dashboard', page_icon=':heartpulse:', layout='wide')
    
    # Set app title
    st.title('Health Prediction App')
    
    # Load and preprocess data
    data = load_and_preprocess_data('MediSync Data.csv')

    # Display distributions
    st.subheader('Data Distributions')
    plot_distributions(data)

    # Extract features and target
    features = ['temperature', 'ecg', 'pulse']
    X = data[features]
    y = data['sleepscore']
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train your model (replace with your actual training code)
    # Example using XGBRegressor (replace with your model)
    model = XGBRegressor()
    model.fit(X_scaled, y)
    
    # Sidebar with app description and inputs
    st.sidebar.subheader('About')
    st.sidebar.text('This app predicts health scores based on input data.')
    
    st.sidebar.subheader('Enter Current Health Data:')
    temperature = st.sidebar.number_input('Temperature', min_value=0.0, max_value=100.0, value=28.1, step=0.1)
    ecg = st.sidebar.number_input('ECG', min_value=0.0, max_value=5.0, value=0.8, step=0.1)
    pulse = st.sidebar.number_input('Pulse', min_value=0, max_value=200, value=82, step=1)
    
    # Button to trigger prediction
    if st.sidebar.button('Predict'):
        # Example data for prediction
        current_data = pd.DataFrame({
            'temperature': [temperature],
            'ecg': [ecg],
            'pulse': [pulse]
        })

        # Predict using the trained model
        predicted_score = predict(model, scaler, current_data)
        
        # Display prediction
        st.subheader('Prediction')
        st.write(f'Predicted Health Score: {predicted_score:.2f}')

        # Calculate percentile rank (example calculation, adjust as needed)
        original_data = load_and_preprocess_data('MediSync Data.csv')
        percentile_rank = np.percentile(original_data['sleepscore'], predicted_score)
        st.write(f'Percentile Rank: {percentile_rank:.2f}')

    # Footer
    st.sidebar.markdown('---')
    st.sidebar.text('Developed by TEAM UNICORN')
    st.sidebar.text('LinkedIn | GitHub | Website')

if _name_ == '_main_':
    main()