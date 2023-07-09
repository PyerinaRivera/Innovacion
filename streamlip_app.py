import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns

data=pd.read_csv('DatosAbiertos_consumohdna_202304.csv', encoding='latin-1' , sep=';')
def main():
  df=load_data()
  page=st.sidebar.selectbox("Seleccione el contenido", ["Inicio", "Diccionario", "Manejo de datos"])
