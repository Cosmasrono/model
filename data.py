import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Load dataset
dataset = pd.read_csv('agricultural_dataset.csv')

# Separate features (X) and target (y)
X = dataset.drop(columns=['label'])
y = dataset['label']

# Train machine learning model
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_scaled, y)

# Function to recommend crop
 
# Function to recommend crop with explanation
# Function to recommend crop with explanation
def recommend_crop(N, P, K, temperature, humidity, ph, rainfall, location, soil_texture):
    # Prepare input features as a numpy array
    input_features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    # Standardize input features
    input_features_scaled = scaler.transform(input_features)
    # Predict crop
    predicted_crop = model.predict(input_features_scaled)
    # Explanation for the recommendation based on location and soil texture
    explanation = f"The recommended crop is {predicted_crop} because it is most suitable based on the provided input conditions, location: {location}, and soil texture: {soil_texture}."
    return predicted_crop, explanation




# Streamlit app
st.title("Crop Recommendation System")

# Define input widgets
N = st.slider('Nitrogen', min_value=0.0, max_value=150.0, value=20.0)
P = st.slider('Phosphorus', min_value=0.0, max_value=150.0, value=20.0)
K = st.slider('Potassium', min_value=0.0, max_value=200.0, value=40.0)
temperature = st.slider('Temperature', min_value=0.0, max_value=50.0, value=10.0)
humidity = st.slider('Humidity', min_value=0.0, max_value=100.0, value=40.0)
ph = st.slider('pH', min_value=0.0, max_value=10.0, value=2.0)
rainfall = st.slider('Rainfall', min_value=10.0, max_value=300.0, value=40.0)
location = st.text_input('Location')
soil_texture = st.selectbox('Soil Texture', ['Sandy', 'Loamy', 'Clay', 'Silt', 'Peat'])

# Button to trigger recommendation
if st.button("Recommend"):
    recommended_crop, explanation = recommend_crop(N, P, K, temperature, humidity, ph, rainfall, location, soil_texture)
    st.write(f"Recommended Crop: {recommended_crop}")
    st.write(explanation)