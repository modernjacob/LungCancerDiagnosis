import streamlit as st
import pandas as pd


st.set_page_config(
    page_title="AI Model Info",
    layout="centered"
)
st.title("AI Model Info")

# MODEL INFO
st.subheader("This model is a deep learning neural network \
             which was train on a datset of 310 patients regarding \
             various symptoms and health information.")
st.write('')
st.subheader('Diagnosis Neural Network', anchor='')
st.image('images/ann_nn.png')
st.write('')

# MODEL PERFORMANCE METRICS
st.subheader('Model Performance')
st.write('The performance of the AI diagnosis model is critical. \
        Important evaluation metrics such as accuracy, precision, recall \
        and F1 score has been analyzed to ensure a optimal performance the \
        classification model. Due to the nature of diagnosising \
        patients, the model should emphasize reducing false negatives \
        and catching true positives.')
st.markdown('**Scores**')
st.write('- Accuracy: 95.83%')
st.write('- Precision: 97.62%')
st.write('- Recall: 93.18%')
st.write('- F1 Score: 0.95')
st.write('')
st.markdown('**Performance Legend**')
st.write('- Accuracy: Estimate of how well the model is likely to perform on \
         unseen data')
st.write('- Precision: Measures the accuracy of positive predictions')
st.write('- Recall: Measures the ability of the model to correctly \
         identify positive cases')
st.write('- F1 Score: Harmonic mean of precision and \
         recall')
st.write('')
st.markdown('**CAUTION:** This AI model is generating a prediction \
         that is based off of its original dataset. The \
         diagnosis outputs of the model should not be \
         treated as conclusive, but as a tool to analyze your \
         patients data against other patients who have tested \
         positive or negative based off of similiar symptoms \
         and health information. The current model\'s threshold for \
         diagnosing a patient as positive is anything above a 0.5 \
         probability prediction.')

st.divider()

# ORIGINAL DATASET
st.subheader('Original Dataset')
ds = pd.read_csv('project_files/data/survey_lung_cancer.csv')
st.dataframe(ds)
st.text('- 1: No')
st.text('- 2: Yes')