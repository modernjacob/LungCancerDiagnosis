# Author: Jacob Huckleberry
# CSU Project Capstone
# 2023.05.20


# DEPENDENCIES
from keras.models import load_model
import pandas as pd
import pickle

# streamlit run project_files/diagnosis.py
# *stop server* ctrl + c
import streamlit as st


#########


# STREAMLIT USER INTERFACE
st.set_page_config(
    page_title="Lung Cancer Diagnosis",
    layout="centered"
)

# UI title
st.title('Lung Cancer Diagnosis')
#st.image('images/lungs-37825_1280.png', width=450
st.text('')
st.text('')

# UI Data input
yes_no = ['Yes', 'No']


with st.container():
    st.text("Enter your patients data to get a suggested diagnosis from the AI model.")
    st.text('')
    patient_id = st.number_input('Patient ID:', min_value=1,max_value=999999, )
    st.text('')
    age = st.slider('Age:', 14, 100)
    st.text('')

    col_1, col_2 = st.columns(2)
    with col_1:
        sex = st.radio('Sex:', ["M", "F"])
        st.text('')
        smoker = st.radio('Smoker:', yes_no)
        st.text('')
        yf = st.radio('Yellow Fingers:', yes_no)
        st.text('')
        anx = st.radio('Anxiety:', yes_no)
        st.text('')
        pressure = st.radio('Peer Pressure:', yes_no)
        st.text('')
        cd = st.radio('Chronic Disease:', yes_no)
        st.text('')
        fatigue = st.radio('Fatigue:', yes_no)
    with col_2:
        allergies = st.radio('Allergies:', yes_no)
        st.text('')
        wheezing = st.radio('Wheezing:', yes_no)
        st.text('')
        alcohol = st.radio('Consumes Alcohol:', yes_no)
        st.text('')
        coughing = st.radio('Cough:', yes_no)
        st.text('')
        breath = st.radio('Shortness of Breath:', yes_no)
        st.text('')
        sd = st.radio('Swallowing Difficulty:', yes_no)
        st.text('')
        cp = st.radio('Chest Pain:', yes_no)

st.text('')
st.text('')
st.text('')


#########


# DATA PIPELINE
@st.cache_data
def data_pipeline(id, sex, age, smoker, yf, anx, pressure, cd, fatigue, allergies, wheezing, alcohol, coughing, breath, sd, cp):

    # transform data dictionary
    data = {
        "PATIENT_ID": [id],
        "GENDER": [sex],
        "AGE": [age],
        "SMOKING": [smoker],
        "YELLOW_FINGERS": [yf],
        "ANXIETY": [anx],
        "PEER_PRESSURE": [pressure],
        "CHRONIC_DISEASE": [cd],
        "FATIGUE": [fatigue],
        "ALLERGY": [allergies],
        "WHEEZING": [wheezing],
        "ALCOHOL_CONSUMING": [alcohol],
        "COUGHING": [coughing],
        "SHORTNESS_OF_BREATH": [breath],
        "SWALLOWING_DIFFICULTY": [sd],
        "CHEST_PAIN": [cp]
    }

    # convert dictionary into pandas dataframe
    db = pd.DataFrame(data)
    df = pd.DataFrame(data)
    df = df.drop('PATIENT_ID', axis=1)
    
    preprocess(df, db)


# DATA SCALING
@st.cache_data
def preprocess(data, db):
    categorical_columns = [
        'SMOKING','YELLOW_FINGERS','ANXIETY','PEER_PRESSURE','CHRONIC_DISEASE',
        'FATIGUE','ALLERGY','WHEEZING','ALCOHOL_CONSUMING','COUGHING',
        'SHORTNESS_OF_BREATH','SWALLOWING_DIFFICULTY','CHEST_PAIN'
    ]

    # Load scaler
    with open('scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)

    # Scale new data
    data['AGE'] = scaler.transform(data[['AGE']])

    # transform categorical columns
    data['GENDER'] = data['GENDER'].replace({"M": 1, "F": 0})
    data[categorical_columns] = data[categorical_columns].replace({"Yes": 1, "No": 0})

    # pass all values to X
    X = data.values

    diagnosis(X, db)


# DIAGNOSIS
@st.cache_data
def diagnosis(data, db):

    database = pd.read_csv("data/database.csv")

    # load diagnosis model and make prediction
    model = load_model("lcd_model.h5")
    diagnosis = model.predict(data)
    
    # diagnosis condition
    if diagnosis > 0.5:
        prediction = "Positive"
        st.error('Diagnosis: Positive')
    else:
        prediction = "Negative"
        st.success('Diagnosis: Negative')
        st.balloons()
    # probability score
    st.write(f'Probability Score: {diagnosis*100}%')

    # append diagnosis to the database
    db["DIAGNOSIS"] = prediction
    add_data = pd.DataFrame(db)

    append_db = pd.concat([database, add_data])
    append_db.to_csv('data/database.csv', index=False)


######


# DIAGNOSIS BUTTON
st.button(
    'Diagnosis', 
    on_click=data_pipeline,
    args = (
        patient_id,
        sex, 
        age, 
        smoker, 
        yf, 
        anx, 
        pressure, 
        cd, 
        fatigue, 
        allergies, 
        wheezing, 
        alcohol, 
        coughing, 
        breath, 
        sd, 
        cp
    )
)
