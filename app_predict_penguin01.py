
import numpy as np
import pandas as pd
import plotly.graph_objects as px
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import streamlit as st

st.title('My ML-Workshop')

tab1, tab2, tab3 = st.tabs(["Penquin Preditiction", "Evaluation", "About us"])

with tab1:
    
    model = pickle.load(open('model.penguins.sav','rb'))
    island_encoder = pickle.load(open('encoder.island.sav','rb'))
    sex_encoder = pickle.load(open('encoder.sex.sav','rb'))
    species_encoder = pickle.load(open('encoder.species.sav','rb'))
    evaluations = pickle.load(open('evals.all.sav','rb'))

    st.header("Penguin Species Prection :)")
    
    x1 = st.radio('select island', island_encoder.classes_)
    x1 = island_encoder.transform([x1])[0]
    # x1
    x2 = st.slider('select culmen length (mm)', 25, 70, 40)
    x3 = st.slider("เลือก culmen depth (mm)", 10,30,15 )
    x4 = st.slider("เลือก flipper length (mm)", 150,250,200)
    x5 = st.slider("เลือก body mass (g)", 2500,6500,3000)
    x6 = st.radio("เลือก sex ",sex_encoder.classes_)
    x6 = sex_encoder.transform([x6])[0]


    x_new = pd.DataFrame(data=np.array([x1, x2, x3, x4, x5, x6]).reshape(1,-1), 
                     columns=['island', 'culmen_length_mm', 'culmen_depth_mm','flipper_length_mm', 'body_mass_g', 'sex'])

    pred = model.predict(x_new)

    st.write('Predicted Species: ' , species_encoder.inverse_transform(pred)[0])
    
with tab2:   
    st.header("Evaluations on Five Techniques")
    evaluations = pickle.load(open('evals.all.sav','rb')) 
    st.dataframe(evaluations)
    
    x = evaluations.columns
    fig = px.Figure(data=[
        px.Bar(name = 'Decision Tree',
               x = x,
               y = evaluations.loc['Decision Tress']),
        px.Bar(name = 'Random Forest',
               x = x,
               y =  evaluations.loc['Random Forest']),
        px.Bar(name = 'KNN',
               x = x,
               y =  evaluations.loc['KNN']),
        px.Bar(name = 'AdaBoost',
               x = x,
               y =  evaluations.loc['AdaBoost']),
        px.Bar(name = 'XGBoost',
               x = x,
               y =  evaluations.loc['XGBoost'])
    ])
    st.plotly_chart(fig, use_container_width=True)
    
 
