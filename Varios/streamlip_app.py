import streamlit as st
import pandas as pd

# Cargar el dataset y almacenarlo en caché
@st.cache
def load_dataset():
    dataset = pd.read_csv('Casos_Anemia_Region_Cusco_2010_2020_Cusco.csv', encoding='latin-1' , sep=';')
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
    st.title("Casos de Anemia por Edades entre los años 2010 - 2020 en la Region de Cusco")
    c1,c2=st.columns([3,7])
    c1.image('cusco1.jpg', width=200)
    c2.markdown("## Modelos predictivos con aprendizaje automático")
    c2.markdown("#### Integrantes:")
    c2.write("- Rivera Cumpa Pyerina")

def show_page1():
    st.title("Carga de datos del dataset")
    st.write("Através de la librería pandas se realiza la carga de datos de nuestro dataset")
    
    # Crear un contenedor con un estilo de fondo personalizado
    contenedor = st.container()
    
    # Cargar el dataset
    st.markdown("### Importar librería")
    st.write("import pandas as pd")
    
    st.markdown("### Cargar datos")
    st.write("""@st.cache
    def load_dataset():
    dataset = pd.read_csv('Casos_Anemia_Region_Cusco_2010_2020_Cusco.csv', encoding='latin-1' , sep=';')
    return dataset""")

    st.markdown("### Mostrar datos")
    dataset = load_dataset()
    # Mostrar la tabla con los datos
    st.write(dataset)

    

def show_page2():
    st.title("Describir datos")
    st.write("Importante para determinar problemas de calidad de datos")
    dataset = load_dataset()
    
    # Mostrar descripción de los datos
    st.markdown("### Descripción del dataset:")
    st.write(dataset.describe())
    st.write("De los resultados obtenemos que: El rango de edades se encuentra entre 0 y 59 años, también que el promedio es de 26,65")
    
    # Obtener el contenido de los datos de las subcategorías de la categoría "provincia"
    subcategorias = dataset["PROVINCIA"].value_counts()
    st.markdown("### Conteo por categoría:")
    st.write("Categoría provincia: ")
    st.write(subcategorias)
    st.write("Los resultados muestran que la moda de la categoría provincia es La Convención")
    
    # Obtener el contenido de los datos de las subcategorías de la categoría "distrito"
    subcategoriaD = dataset["DISTRITO"].value_counts()
    st.markdown("### Conteo por categoría:")
    st.write("Categoría provincia: ")
    st.write(subcategoriaD)
    st.write("Los resultados muestran que la moda de la categoría distrito es La Convención")


def show_page3():
    st.title("Página 3")
    st.write("Esta es la página 3.")
    

if __name__ == "__main__":
    main()

