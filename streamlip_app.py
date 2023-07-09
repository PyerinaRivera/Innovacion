import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns

image = Image.open('descarga.png')
st.image(image, caption='',use_column_width=True)

st.title("Test de riesgo Covid-19 :sunglasses:")
data=pd.read_csv('DatosAbiertos_consumohdna_202304.csv', encoding='latin-1' , sep=';')
