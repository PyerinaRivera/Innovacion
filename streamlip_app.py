import streamlit as st

st.sidebar.write('Siderbar')
if st.siderbar.button('opcion 1'):
  c1,c2=st.columns([3,7])
  c1.image('descarga.png',width=200)
  c2.markdown("## Dataset: Consumo energético de clientes Hidrandina [Distriliuz - DLZ]")
  c2.markdown("### Integrantes:")
  c2.markdown("###### - Rivera Cumpa Pyerina")
