import streamlit as st

st.set_page_config(
    page_title="Avance 3",
    page_icon="💻",
)

def main():
    # Configurar la barra de navegación
    st.sidebar.title("Navegación")
    pages = {
        "Inicio": show_home,
        "Página 1": show_page1,
        "Página 2": show_page2,
        "Página 3": show_page3
    }
    page = st.sidebar.selectbox("Ir a", tuple(pages.keys()))

    # Mostrar la página seleccionada
    pages[page]()

def show_home():
    st.title ("Dataset: Consumo energético de clientes Hidrandina [Distriliuz - DLZ]")
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

def show_page3():
    st.title("Página 3")
    st.write("Esta es la página 3.")

if __name__ == "__main__":
    main()

