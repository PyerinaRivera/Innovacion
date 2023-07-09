import streamlit as st

st.set_page_config(
    page_title="Avance 3",
    page_icon="💻",
)

def main():
    # Configurar la barra de navegación
    st.sidebar.title("Navegación")
    pages = {
        "Inicio": home,
        "Cargar": pagina1,
        "Describir": pagina2
        "Visualizar": pagina3
    }
    page = st.sidebar.selectbox("Ir a", tuple(pages.keys()))

    # Mostrar la página seleccionada
    pages[page]()

def show_home():
    st.write("# Dataset: Consumo energético de clientes Hidrandina [Distriliuz - DLZ]")
	st.markdown(
  	"""Avance 3: Modelos predictivos con aprendizaje automático
  	### Integrantes:
  	- Rivera Cumpa Pyerina
  	""")
def show_page1():
    st.title("Página 1")
    st.write("Esta es la página 1.")

def show_page2():
    st.title("Página 2")
    st.write("Esta es la página 2.")

if __name__ == "__main__":
    main()
