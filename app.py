import pickle

import streamlit as st
import pandas as pd
import numpy as np
import sklearn

st.write("""
# Hear Disease prediction app
This app predicts if a person is likely to have a stroke or not
The data is obtained from kaggle uploaded by david Lapp
""")

st.sidebar.header('User Input Features')

st.sidebar.markdown("""
Example CSV input: 
""")

uploaded_file = st.sidebar.file_uploader("Upload the input CSV file", type=["csv"])
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)

else:
    def user_features():
        age = st.sidebar.slider("Age", 29.00, 77.00, 54.43)
        sex = st.sidebar.selectbox("Gender, Male 1\nFemale 0 ", (0, 1))
        cp = st.sidebar.selectbox(
            "Chest Pain type: 0 Typical Angima\n1: Atypical Angima\n2: Non-anginal Pain\n3: Asymptomatic", (0, 1, 2, 3))
        trestbps = st.sidebar.slider("Resting Blood Pressure", 94, 200, 100)
        chol = st.sidebar.slider("Cholesterol", 126, 564, 300)
        fbs = st.sidebar.slider("Fasting Blood Sugar", 0.00, 1.00, 0.34)
        restecg = st.sidebar.selectbox('Resting ECG', (0, 1, 2))
        thalach = st.sidebar.slider("Max heart rate achieved", 202, 71, 149)
        exang = st.sidebar.selectbox("Exercise induced angia", (0, 1))
        oldpeak = st.sidebar.slider("Oldpeak, ST depression induced by exercise relative to rest", 6.2, 0.00, 1.07)
        slope = st.sidebar.selectbox("The slope of the peak exercise ST segment", (2, 0, 1))
        ca = st.sidebar.selectbox("Number of major vessels (0-4) colored by flourosopy", (0, 1, 2, 3, 4))
        thal = st.sidebar.selectbox("Thal:\n0 = normal\n1 = fixed defect\n2 = Reversible defect", (0, 1, 2))
        data = {
            'age': age,
            'sex': sex,
            'cp': cp,
            'trestbps': trestbps,
            'chol': chol,
            'fbs': fbs,
            'restecg': restecg,
            'thalach': thalach,
            'exang': exang,
            'oldpeak': oldpeak,
            'slope': slope,
            'ca': ca,
            'thal':thal
        }
        features = pd.DataFrame(data,index=[0])
        return features
    input_df = user_features()

st.subheader('User Input Features')

if uploaded_file is not None:
    st.write(input_df)
else:
    st.write("Awaiting input file, currently using example info")
    st.write(input_df)

load_clf = pickle.load(open('HeartClf.pkl','rb'))
prediction=load_clf.predict(input_df)
pred_prob = load_clf.predict_proba(input_df)

st.subheader("Prediction")

if(prediction==1):
    st.write("High Probability")
else:
    st.write("Low probabilit")

st.subheader("Prediction Probability")
st.write(pred_prob)
