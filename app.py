from ossaudiodev import openmixer
import numpy as np
import pandas as pd
import streamlit as st
import pickle
from PIL import Image
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv('OPENAI_API_KEY')
)

def generate_autocompletion(prompt):
    crop_role = f""" To receive a specific crop recommendation, please provide details such as your location, soil type, climate, etc.

 """

    # Call the OpenAI API
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": crop_role}, {"role": "user", "content": prompt}],
    )
    # Extract the output text from the API response
    output_text = response.choices[0].message.content
    print(f'Output text: {response.choices[0].message.content}')
    return st.markdown(output_text)



st.set_page_config(
    page_title="Recommendation System for Agriculture",
    page_icon=":random:",
    layout="centered",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "# This application will help provide crop recommendations!"
    }
)

model = pickle.load(open('crop_model.pkl','rb'))

ss = pickle.load(open('standardscaler.pkl','rb'))
ms = pickle.load(open('minmaxscaler.pkl','rb'))

check_crops = {1: 'rice',
 2: 'maize',
 3: 'jute',
 4: 'cotton',
 5: 'coconut',
 6: 'papaya',
 7: 'orange',
 8: 'apple',
 9: 'muskmelon',
 10: 'watermelon',
 11: 'grapes',
 12: 'mango',
 13: 'banana',
 14: 'pomegranate',
 15: 'lentil',
 16: 'blackgram',
 17: 'mungbean',
 18: 'mothbeans',
 19: 'pigeonpeas',
 20: 'kidneybeans',
 21: 'chickpea',
 22: 'coffee',
 21: 'chickpea',
   22: 'coffee', 23: 'corn',
    24: 'potato', 25: 'tomato', 26: 'wheat', 27: 'barley', 28: 'sunflower', 29: 'sugarbeet', 30: 'carrot',
    31: 'cabbage', 32: 'cauliflower', 33: 'lettuce', 34: 'spinach', 35: 'celery', 36: 'asparagus', 37: 'bellpepper',
    38: 'chili', 39: 'eggplant', 40: 'onion', 41: 'garlic', 42: 'ginger', 43: 'turmeric', 44: 'radish', 45: 'turnip',
    46: 'sweetpotato', 47: 'peanut', 48: 'soybean', 49: 'bean', 50: 'pea', 51: 'zucchini', 52: 'squash', 53: 'melon',
    54: 'cucumber', 55: 'broccoli', 56: 'kale', 57: 'brusselsprouts', 58: 'beetroot', 59: 'parsnip', 60: 'leek',
    61: 'fennel', 62: 'artichoke', 63: 'okra', 64: 'rhubarb', 65: 'kohlrabi', 66: 'endive', 67: 'arugula',
    68: 'watercress', 69: 'chives', 70: 'coriander', 71: 'dill', 72: 'basil', 73: 'mint', 74: 'oregano', 75: 'rosemary',
    76: 'sage', 77: 'thyme', 78: 'parsley', 79: 'lavender', 80: 'cilantro'}

def recommend(N, P, K, temperature, humidity, ph, rainfall):
    features = np.array([[N, P, K, temperature, humidity, ph, rainfall]]).reshape(1,-1)
    features = ms.transform(features)
    features = ss.transform(features)
    prediction = model.predict(features)
    return prediction[0]

def output(N, P, K, temperature, humidity, ph, rainfall):
    predict = recommend(N, P, K, temperature, humidity, ph, rainfall)
    if predict in check_crops:
        crop = check_crops[predict]
        st.write("""# Our crop recommendation is """, crop)
        with st.spinner('Getting best crop advice...'):
            autocompletion = generate_autocompletion(crop)
    else:
        st.write("""# No recommendation""")



      
 

st.write("The mean values of input variables are provided in the above table. Refer the above table to set the input variables and see the accuracy of the recommendation!")

with st.sidebar:
    image = Image.open('./sidebar_image.jpg')
    st.image(image)
    st.markdown("<h2 style='text-align: center; color: red;'>Settings Tab</h2>", unsafe_allow_html=True)


    st.write("Input Settings:")

    #define the N for the model
    n_value = st.slider('Nitrogen :', 0.0, 150.0, 20.0)

    #define the P for the model
    p_value = st.slider('Phosphorus :', 0.0, 150.0, 20.0)

    #define the K for the model
    k_value = st.slider('pottasium  :', 0.0, 200.0, 40.0)

    #define the temperature for the model
    temperature = st.slider('Temperature :', 0.0, 50.0, 10.0)

    #define the humidity for the model
    humidity = st.slider('Humidity  :', 0.0, 100.0, 40.0)

    #define the ph for the model
    ph_value = st.slider('pH  :', 0.0, 10.0, 2.0)

    #define the rainfall for the model
    rainfall = st.slider('Rainfall  :', 10.0, 300.0, 40.0)


with st.container():
    if st.button("Recommend"):
        output(n_value, p_value, k_value, temperature, humidity, ph_value, rainfall)