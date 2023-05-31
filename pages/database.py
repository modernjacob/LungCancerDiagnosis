import csv
import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Database",
    layout="wide"
)

st.title("Database")

# get database
database = pd.read_csv('data/database.csv')

st.write(database, use_container_width=True)

def clear_db():
    columns = [
        "PATIENT_ID",
        "GENDER",
        "AGE",
        "SMOKING",
        "YELLOW_FINGERS",
        "ANXIETY",
        "PEER_PRESSURE",
        "CHRONIC_DISEASE",
        "FATIGUE",
        "ALLERGY",
        "WHEEZING",
        "ALCOHOL_CONSUMING",
        "COUGHING",
        "SHORTNESS_OF_BREATH",
        "SWALLOWING_DIFFICULTY",
        "CHEST_PAIN",
        "DIAGNOSIS"
    ]
    with open('data/database.csv', 'w') as file:
        writer = csv.writer(file)
        writer.writerow(columns)

st.write('')

st.button(
    'Clear Database',
    on_click=clear_db
)

st.write('')

# download button
st.download_button('Download CSV', database.to_csv(), file_name='diagnosis_data.csv')
