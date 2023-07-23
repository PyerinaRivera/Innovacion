import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Cargar el dataset y almacenarlo en caché
@st.cache
def load_dataset():
    dataset = pd.read_csv('Casos_Anemia_Region_Cusco_2010_2020_Cusco.csv', encoding='latin-1', sep=';')
    return dataset

def main():
    # Configurar la barra de navegación
    st.sidebar.title("Navegación")
    pages = {
        "Inicio": show_home,
        "Cargar": show_page1,
        "Describir": show_page2,
        "Visualizar": show_page3,
        "Diccionario": show_page4,
        "Modelo Predictivo": show_page5,
        "Modelo Predictivo 2": show_page6
    }
    page = st.sidebar.selectbox("Ir a", tuple(pages.keys()))

    # Mostrar la página seleccionada
    pages[page]()

def show_home():
    st.title("Casos de Anemia por Edades entre los años 2010 - 2020 en la Región de Cusco")
    st.image('cusco1.jpg', width=200)
    st.markdown("## Modelos predictivos con aprendizaje automático")
    st.markdown("#### Integrantes:")
    st.write("- Rivera Cumpa Pyerina")
    st.write("- Gavilan Guevara Marius Randy")
    st.write("- Lopez Peña Fritz Alexander")
    st.write("- Moron Espinoza Luis Fernando")
    st.write("- Almanacin Soncco Edgar")

def show_page1():
    st.title("Carga de datos del dataset")
    st.write("A través de la librería pandas se realiza la carga de datos de nuestro dataset")
    
    # Cargar el dataset
    st.markdown("### Importar librería")
    st.write("import pandas as pd")
    
    st.markdown("### Cargar datos")
    st.code("""@st.cache
def load_dataset():
    dataset = pd.read_csv('Casos_Anemia_Region_Cusco_2010_2020_Cusco.csv', encoding='latin-1', sep=';')
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
    st.write("De los resultados obtenidos, se puede observar que el rango de edades se encuentra entre 0 y 59 años, también que el promedio es de 26.65.")
    
    # Obtener el contenido de los datos de las subcategorías de la categoría "provincia"
    subcategorias = dataset["PROVINCIA"].value_counts()
    st.markdown("### Conteo por categoría:")
    st.write("Categoría provincia: ")
    st.write(subcategorias)
    st.write("Los resultados muestran que la moda de la categoría provincia es La Convención.")
    
    # Obtener el contenido de los datos de las subcategorías de la categoría "distrito"
    subcategoriaD = dataset["DISTRITO"].value_counts()
    st.write("Categoría distrito: ")
    st.write(subcategoriaD)
    st.write("Los resultados muestran que la moda de la categoría distrito es Echarate.")

    subcategoriaA = dataset["ANIO"].value_counts()
    st.write("Categoría año: ")
    st.write(subcategoriaA)
    st.write("Los resultados muestran que la moda de la categoría año es 2019.")

    st.markdown("### Promedio por categoría:")
    # Calcular el promedio de edad por provincia
    promedio_edad_por_provincia = dataset.groupby("PROVINCIA")["EDAD"].mean()
    # Mostrar el promedio de edad por provincia
    st.write("Promedio de edad por provincia:")
    st.write(promedio_edad_por_provincia)
    st.write("De los resultados obtenidos, se puede observar que los promedios de edad por provincia se encuentran en un rango de 20 y 30 años.")

    # Seleccionar las provincias a comparar
    st.write("Promedio de casos de anemia por año y provincia")
    provincias = ['ESPINAR', 'CANAS', 'PARURO']
    # Filtrar los datos para las provincias seleccionadas
    data_provincias = dataset[dataset['PROVINCIA'].isin(provincias)]
    # Calcular los promedios de casos de anemia por año y provincia
    promedios_por_anio_provincia = data_provincias.groupby(['ANIO', 'PROVINCIA'])['CASOS'].mean().unstack()
    st.write(promedios_por_anio_provincia)
    st.write("De los resultados podemos observar cuál es el promedio de casos de anemia que se registraron en cada provincia en un determinado año, por ejemplo en el año 2015 en la provincia de Espinar se registraron en promedio 3.2 casos de anemia.")

def show_page3():
    st.title("Visualizar datos: ")
    dataset = load_dataset()

    st.write("Los gráficos presentados a continuación tienen el objetivo de mostrar pictóricamente los datos que se tienen en el dataset para una mejor comprensión.")
    # Calcular el promedio de edad por provincia
    st.markdown("### Gráficos de barras:")
    st.write("Gráfico de barras del promedio de edad por provincia de los pacientes con anemia en Cusco:")
    promedio_EP = dataset.groupby('PROVINCIA')['EDAD'].mean()
    st.bar_chart(promedio_EP)
    
    # Conteo de los datos por provincia
    st.write("Gráfico de barras del número de casos de anemia por provincia:")
    conteo = dataset["PROVINCIA"].value_counts()
    st.bar_chart(conteo)

    st.write("Gráfico de barras del promedio de casos de anemia por año y provincia")
    # Seleccionar las provincias a comparar
    provincias = ['ESPINAR', 'CANAS', 'PARURO']
    # Filtrar los datos para las provincias seleccionadas
    data_provincias = dataset[dataset['PROVINCIA'].isin(provincias)]
    # Calcular los promedios de casos de anemia por año y provincia
    promedios_por_anio_provincia = data_provincias.groupby(['ANIO', 'PROVINCIA'])['CASOS'].mean().unstack()
    # Mostrar el gráfico de barras
    st.bar_chart(promedios_por_anio_provincia)

    st.write("Gráfico de barras de la relación entre los casos de anemia total y casos de anemia normal por provincia")
    promedio = dataset.groupby('PROVINCIA').agg({'CASOS': 'sum', 'NORMAL': 'sum'})
    plt.figure(figsize=(16, 6))
    promedio.plot(kind='bar')
    # Utiliza el método st.pyplot() para mostrar el gráfico en Streamlit
    st.pyplot(plt.gcf())

    # GRAFICO DE BARRAS DE CANTIDAD DE CASOS(Numero de casos con anemia por debajo del indicador de salud) Y NORMAL(Numero de casos en condiciones normales (sin anemia)) POR DISTRITO DE LA PROVINCIA DE CHUMBIVILCAS
    st.write("Gráfico de barras de la relación entre los casos de anemia total y casos de anemia normal por distritos de la provincia de Chumbivilcas")
    datos_prov = dataset[dataset['PROVINCIA'] == 'CHUMBIVILCAS']
    promedio = datos_prov.groupby('DISTRITO').agg({'CASOS': 'sum', 'NORMAL': 'sum'})
    plt.figure(figsize=(16, 6))
    promedio.plot(kind='bar')
    plt.title('CANTIDAD DE CASOS POR DISTRITO DE LA PROVINCIA DE CHUMBIVILCAS')
    st.pyplot(plt.show())

    st.write("Gráfico de promedio de casos de anemia por año y provincia")
    provincias = ['ESPINAR', 'CANAS', 'PARURO']
    # Filtrar los datos para las provincias seleccionadas
    data_provincias = dataset[dataset['PROVINCIA'].isin(provincias)]
    # Calcular los promedios de casos de anemia por año y provincia
    promedios_por_anio_provincia = data_provincias.groupby(['ANIO', 'PROVINCIA'])['CASOS'].mean().unstack()
    # Crear el gráfico de barras para los promedios
    fig, ax = plt.subplots(figsize=(10, 6))
    promedios_por_anio_provincia.plot(kind='bar', ax=ax)
    plt.title('Promedio de casos de anemia por año y provincia')
    plt.xlabel('Año')
    plt.ylabel('Promedio de casos')
    plt.legend(title='Provincia')
    # Utiliza el método st.pyplot() para mostrar el gráfico en Streamlit
    st.pyplot(fig)
    
    st.markdown("### Gráficos de líneas:")
    st.write("Gráfico de la evolución de casos de anemia por provincia")
    # Seleccionar las provincias a comparar
    provincias = ['CUSCO', 'CALCA', 'ANTA']
    # Filtrar los datos para las provincias seleccionadas
    data_provincias = dataset[dataset['PROVINCIA'].isin(provincias)]
    # Agrupar los datos por año y provincia y calcular el total de casos por año
    casos_por_anio_provincia = data_provincias.groupby(['ANIO', 'PROVINCIA'])['CASOS'].sum().unstack()
    # Mostrar el gráfico de líneas múltiples en un solo gráfico
    for provincia in provincias:
        st.line_chart(casos_por_anio_provincia[provincia])
        
    st.write("Gráfico de líneas unificado de casos de anemia en la provincia Cusco, Calca y Anta")
    # Seleccionar las provincias a comparar
    provincias = ['CUSCO', 'CALCA', 'ANTA']
    # Filtrar los datos para las provincias seleccionadas
    data_provincias = dataset[dataset['PROVINCIA'].isin(provincias)]
    # Agrupar los datos por año y provincia y calcular el total de casos por año
    casos_por_anio_provincia = data_provincias.groupby(['ANIO', 'PROVINCIA'])['CASOS'].sum().unstack()
    # Crear el gráfico de líneas múltiples
    plt.figure(figsize=(10, 6))
    for provincia in provincias:
        plt.plot(casos_por_anio_provincia.index, casos_por_anio_provincia[provincia], marker='o', label=provincia)
    plt.title('Evolución de casos de anemia por provincia')
    plt.xlabel('Año')
    plt.ylabel('Total de casos')
    plt.legend()
    plt.grid(True)
    st.pyplot(plt.show()) 

    st.markdown("### Gráficos circulares:")
    # En este ejemplo, estamos contando el número de casos de anemia por año y luego creando un gráfico circular que muestra la distribución de casos entre los diferentes años.
    st.write("Gráfico circular número de casos de anemia por año")
    casos_por_año = dataset['ANIO'].value_counts()
    # Crear el gráfico circular
    plt.figure(figsize=(8, 8))
    plt.pie(casos_por_año, labels=casos_por_año.index, autopct='%1.1f%%')
    plt.title('Distribución de casos de anemia por año')
    plt.axis('equal')
    st.pyplot(plt.show())

    # En este ejemplo, estamos contando el número de casos de anemia por microred y creando un gráfico circular que muestra la distribución de casos entre las microredes.
    st.write("Gráfico circular número de casos de anemia por microred")
    casos_por_microred = dataset['MICRORED'].value_counts()
    # Crear el gráfico circular
    plt.figure(figsize=(15, 15))
    plt.pie(casos_por_microred, labels=casos_por_microred.index, autopct='%1.1f%%')
    plt.title('Distribución de casos de anemia por microred')
    plt.axis('equal')
    st.pyplot(plt.show())

    # En este ejemplo, agrupamos los datos por provincia y calculamos el número total de casos de anemia y el promedio de edad de los casos en cada provincia.
    # Luego, ordenamos las provincias por el número total de casos en orden descendente.
    # Creamos un gráfico de barras agrupadas donde las barras representan el número total de casos de anemia por provincia.
    # Además, agregamos una línea punteada que representa el promedio de edad de los casos en cada provincia.
    # El eje y izquierdo corresponde al número de casos, y el eje y derecho corresponde al promedio de edad.
    # Utilizamos colores diferentes para cada eje y ajustamos los parámetros para mostrar las etiquetas de las provincias de forma adecuada.
    st.markdown("### Gráfico de categoría múltiple:")
    st.write("Gráfico de número total de casos y promedio de edad por provincia")
    # Agrupar los datos por provincia y calcular el número total de casos y el promedio de edad
    datos_provincia = dataset.groupby('PROVINCIA').agg({'CASOS': 'sum', 'EDAD': 'mean'})
    # Ordenar las provincias por el número total de casos en orden descendente
    datos_provincia = datos_provincia.sort_values(by='CASOS', ascending=False)
    # Crear el gráfico de barras agrupadas
    fig, ax1 = plt.subplots(figsize=(20, 6))
    # Barra para el número total de casos
    ax1.bar(datos_provincia.index, datos_provincia['CASOS'], color='tab:blue')
    ax1.set_ylabel('Número de casos', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.set_xlabel('Provincia')
    # Promedio de edad como línea punteada
    ax2 = ax1.twinx()
    ax2.plot(datos_provincia.index, datos_provincia['EDAD'], color='tab:red', linestyle='--')
    ax2.set_ylabel('Promedio de edad', color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')
    # Ajustar el espaciado de las etiquetas de las provincias
    plt.xticks(rotation=45, ha='right')
    plt.title('Número total de casos y promedio de edad por provincia')
    st.pyplot(plt.show()) 


    # En este ejemplo, estamos utilizando los datos de edad, casos normales y casos totales para crear un gráfico de dispersión tridimensional.
    # Cada punto en el gráfico representa una combinación de edad, casos normales y casos totales.
    st.markdown("### Diagrama de dispersión:")
    st.write("Relación entre edad, casos normales y casos totales")
    # Obtener los valores de edad, casos normales y casos totales
    edad = dataset['EDAD']
    casos_normales = dataset['NORMAL']
    casos_totales = dataset['TOTAL']
    # Crear el gráfico de dispersión tridimensional
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(edad, casos_normales, casos_totales)
    ax.set_title('Relación entre edad, casos normales y casos totales')
    ax.set_xlabel('Edad')
    ax.set_ylabel('Casos Normales')
    ax.set_zlabel('Casos Totales')
    st.pyplot(plt.show()) 

    st.markdown("### Diagrama de caja y bigotes:")
    st.write("Distribución de edades")
    # Obtener los valores de edad
    edades = dataset['EDAD']
    # Crear el gráfico de caja y bigotes
    plt.figure(figsize=(8, 6))
    plt.boxplot(edades, vert=False)
    plt.title('Distribución de edades')
    plt.xlabel('Edad')
    st.pyplot(plt.show()) 



def show_page4():
    st.title("Casos de Anemia por Edades entre los años 2010 - 2020 en la Región de Cusco [Gobierno Regional Cusco]")
    df = pd.DataFrame(
        [
            {"Variable": "DEPARTAMENTO", "Descripcion": "Nombre del departamento de ubicación del Gobierno Regional de Cusco", "Tipo de dato": "Texto", "Tamaño": 30,},
            {"Variable": "PROVINCIA", "Descripcion": "Nombre de la provincia de ubicación del Gobierno Regional de Cusco", "Tipo de dato": "Texto", "Tamaño": 30,},
            {"Variable": "DISTRITO", "Descripcion": "Nombre del distrito de ubicación del Gobierno Regional de Cusco", "Tipo de dato": "Texto", "Tamaño": 30,},
            {"Variable": "RED", "Descripcion": "Nombre de la red asistencial según organización del Ministerio de Salud", "Tipo de dato": "Texto", "Tamaño": 100,},
            {"Variable": "MICRORED", "Descripcion": "Nombre de la Micro red asistencial de la red de salud", "Tipo de dato": "Texto", "Tamaño": 100,},
            {"Variable": "COD_EESS", "Descripcion": "Codigo del Establecimiento de Salud", "Tipo de dato": "Numérico", "Tamaño": 10,},
            {"Variable": "EESS", "Descripcion": "Nombre del Establecimiento de Salud", "Tipo de dato": "Texto", "Tamaño": 100,},
            {"Variable": "EDAD", "Descripcion": "Edad del grupo de personas diagnosticadas", "Tipo de dato": "Numérico", "Tamaño": 3,},
            {"Variable": "AÑO", "Descripcion": "Año de recopilación de la información", "Tipo de dato": "Numérico", "Tamaño": 4,},
            {"Variable": "CASOS", "Descripcion": "Numero de casos con anemia por debajo del indicador de salud", "Tipo de dato": "Numérico", "Tamaño": 5,},
            {"Variable": "NORMAL", "Descripcion": "Numero de casos en condiciones normales (sin anemia)", "Tipo de dato": "Numérico", "Tamaño": 5,},
            {"Variable": "TOTAL", "Descripcion": "Suma total de casos con anemia y los casos en condiciones normales", "Tipo de dato": "Numérico", "Tamaño": 5,},
            {"Variable": "FECHA_CORTE", "Descripcion": "Fecha de corte de información", "Tipo de dato": "Fecha", "Tamaño": 8,},
            {"Variable": "UBIGEO", "Descripcion": "Codigo de ubicación según INEI", "Tipo de dato": "Texto", "Tamaño": 6,},
        ]
    )
    st.dataframe(df, use_container_width=True)

def preprocess_data(dataset):
    # Eliminamos filas con valores faltantes en la columna 'PROVINCIA'
    dataset = dataset.dropna(subset=['PROVINCIA'])

    # Codificamos las variables categóricas 'PROVINCIA' y 'DISTRITO' utilizando LabelEncoder
    label_encoder = LabelEncoder()
    dataset['PROVINCIA'] = label_encoder.fit_transform(dataset['PROVINCIA'])
    dataset['DISTRITO'] = label_encoder.fit_transform(dataset['DISTRITO'])

    # Codificamos las variables categóricas utilizando OneHotEncoder
    onehot_encoder = OneHotEncoder()
    encoded_columns = pd.DataFrame(onehot_encoder.fit_transform(dataset[['PROVINCIA', 'DISTRITO']]).toarray(),
                                   columns=onehot_encoder.get_feature_names(['PROVINCIA', 'DISTRITO']))
    dataset = pd.concat([dataset, encoded_columns], axis=1)
    
    # Eliminamos las columnas originales 'PROVINCIA' y 'DISTRITO'
    dataset.drop(columns=['PROVINCIA', 'DISTRITO'], inplace=True)
    
    # Verificamos y manejamos valores faltantes si es necesario
    # Por ejemplo, podemos llenar los valores faltantes con la media de la columna
    dataset.fillna(dataset.mean(), inplace=True)

    return dataset

def show_page5():
    st.title("Modelo Predictivo de Casos de Anemia")
    dataset = load_dataset()

    st.write("En esta sección, desarrollaremos un modelo predictivo de casos de anemia utilizando aprendizaje automático.")
    st.write("El primer paso es preparar los datos para el modelado.")
    
    # Preprocesar los datos
    dataset = preprocess_data(dataset)

    # Dividir los datos en características (X) y etiquetas (y)
    X = dataset.drop(columns=['CASOS'])
    y = dataset['CASOS']

    # Dividir los datos en conjuntos de entrenamiento y prueba (80% entrenamiento, 20% prueba)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Crear el modelo de regresión lineal
    model = LinearRegression()

    # Entrenar el modelo utilizando los datos de entrenamiento
    model.fit(X_train, y_train)

    # Evaluar el modelo utilizando los datos de prueba
    score = model.score(X_test, y_test)

    st.write("El modelo ha sido entrenado y evaluado con éxito.")

    st.markdown("### Evaluación del Modelo")
    st.write(f"Coeficiente de determinación (R^2): {score:.2f}")

    st.markdown("### Predicciones")
    st.write("Ingrese los datos para realizar una predicción.")
    edad = st.number_input("Edad del grupo de personas diagnosticadas:", min_value=0, max_value=100, value=30)
    provincia = st.selectbox("Provincia de ubicación del Gobierno Regional de Cusco:", dataset['PROVINCIA'].unique())
    distrito = st.selectbox("Distrito de ubicación del Gobierno Regional de Cusco:", dataset['DISTRITO'].unique())

    # Codificar la provincia y distrito seleccionados
    label_encoder = LabelEncoder()
    dataset['PROVINCIA'] = label_encoder.fit_transform(dataset['PROVINCIA'])
    dataset['DISTRITO'] = label_encoder.fit_transform(dataset['DISTRITO'])
    provincia_encoded = label_encoder.transform([provincia])[0]
    distrito_encoded = label_encoder.transform([distrito])[0]

    # Realizar la predicción utilizando el modelo entrenado
    prediction = model.predict([provincia_encoded, distrito_encoded, edad])

    st.write(f"La predicción de casos de anemia es: {prediction[0]:.2f}")

def entrenar_mmodelo():
    data_nn = load_dataset()
    data_nn = data_nn.groupby('ANIO').agg({'CASOS': 'sum', 'NORMAL': 'sum'})
    data_nn = data_nn.reset_index()

    X = data_nn['ANIO'].values.reshape(-1, 1)
    y = data_nn['CASOS'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    modelo = LinearRegression()

    modelo.fit(X_train, y_train)

    y_pred = modelo.predict(X_test)

    return modelo

modelo_entrenado = entrenar_mmodelo()

def predecircasos(anio):
    casos_anemia_predichos = modelo_entrenado.predict([[anio]])
    return "Predicción de casos de anemia para el año", anio, ":", casos_anemia_predichos[0]
def show_page6():
    st.title("Modelo Predictivo de Casos de Anemia")
    dataset = load_dataset()

    st.write("En esta sección, desarrollaremos un modelo predictivo de casos de anemia utilizando aprendizaje automático.")
    anio = st.number_input("Ingrese el anio:", min_value=2020, max_value=2035, value=2023)
    if st.button('Predecir'):
        # Acciones que se ejecutarán cuando el botón sea presionado
        st.write(predecircasos(anio))

if __name__ == "__main__":
    main()
