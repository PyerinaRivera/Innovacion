import streamlit as st
from streamlit_option_menu import option_menu

st.set_page_config(
    page_title="Avance 3",
    page_icon="💻",
)

with st.siderbar:
	selected = option_menu(
		menu_title="Menú", 
		options=["Inicio","Carga","Describir","Visualizar"],)


st.write("# Dataset: Consumo energético de clientes Hidrandina [Distriliuz - DLZ]")
st.markdown(
  """Avance 3: Modelos predictivos con aprendizaje automático
  ### Integrantes:
  - Rivera Cumpa Pyerina
  """)
