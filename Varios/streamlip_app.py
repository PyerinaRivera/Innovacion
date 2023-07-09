import streamlit as st
from streamlit_option_menu import option_menu

st.set_page_config(
    page_title="Avance 3",
    page_icon="💻",
)

st.siderbar:
	selected = option_menu(
		menu_title="Menú", 
		options=["Inicio","Carga","Describir","Visualizar"],)
if selected=="Inicio":
	st.write("# Dataset: Consumo energético de clientes Hidrandina [Distriliuz - DLZ]")
	st.markdown(
  	"""Avance 3: Modelos predictivos con aprendizaje automático
  	### Integrantes:
  	- Rivera Cumpa Pyerina
  	""")
if selected=="Carga":
	st.tittle("Aquí")
if selected=="Describir":
	st.tittle("Aquí Descr")
if selected=="Visualizar":
	st.tittle("Aquí V")
