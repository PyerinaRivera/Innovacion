import streamlit as st
import pandas as pd

# Cargar el dataset y almacenarlo en caché
@st.cache
def load_dataset():
    dataset = pd.read_csv("")
    return dataset

def main():
    # Configurar la barra de navegación
    st.sidebar.title("Navegación")
    pages = {
        "Inicio": show_home,
        "Cargar": show_page1,
        "Describir": show_page2,
        "Visualizar": show_page3
    }
    page = st.sidebar.selectbox("Ir a", tuple(pages.keys()))

    # Mostrar la página seleccionada
    pages[page]()

def show_home():
    st.title("Dataset: Consumo energético de clientes Hidrandina [Distriliuz - DLZ]")
    c1,c2=st.columns([3,7])
    c1.image('descarga.png', width=200)
    c2.markdown("## Modelos predictivos con aprendizaje automático")
    c2.markdown("#### Integrantes:")
    c2.write("- Rivera Cumpa Pyerina")

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

