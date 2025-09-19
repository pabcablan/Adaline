# 🧠 Adaline (Adaptive Linear Neuron)

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green)


## 📝 Descripción general

Este proyecto contiene una implementación de **Adaline (Adaptive Linear Neuron)**, una de las redes neuronales pioneras de una sola capa.

El objetivo principal es demostrar su capacidad para resolver problemas de clasificación binaria, aprendiendo a trazar una frontera de decisión lineal que separe dos clases de datos. La lógica del modelo está encapsulada en una clase de Python, y su rendimiento se prueba y visualiza en un Jupyter Notebook.


## 📜 Explicación Teórica

**ADALINE (ADAptive LInear NEuron)** es una red neuronal de una sola capa que calcula una salida lineal combinando las entradas `X` con sus pesos `w` y un sesgo `b`:

`y = wᵀ · X + b`

A diferencia del Perceptrón, su método de aprendizaje se basa en:

-   **Salida Continua para el Error:** Utiliza la salida lineal (sin cuantizar) para calcular el error, lo que permite un ajuste más preciso.
-   **Función de Coste (MSE):** Minimiza el **Error Cuadrático Medio** para optimizar el modelo.
-   **Regla de Aprendizaje (Widrow-Hoff):** Actualiza los pesos mediante el **descenso de gradiente**, haciendo que el ajuste sea proporcional al tamaño del error.

Este enfoque permite a Adaline ajustar sus pesos incluso cuando la clasificación es correcta, logrando una convergencia más estable y robusta que la del Perceptrón.


## 📂 Estructura del proyecto

El repositorio se organiza de la siguiente manera:

| Archivo/Directorio | Descripción |
| :--- | :--- |
| [`src/Adaline.py`](src/Adaline.py) | Módulo de Python que contiene la implementación de la clase `Adaline`. |
| [`test/AdalineTester.ipynb`](test/AdalineTester.ipynb) | Jupyter Notebook para probar, entrenar y visualizar el rendimiento del modelo Adaline. |
| [`LICENSE`](LICENSE) | Archivo de licencia del proyecto (MIT). |
| [`README.md`](README.md) | Documentación del proyecto |

## 📦 Requisitos

Para ejecutar este proyecto, necesitas un entorno con **Python 3.8+** y las siguientes librerías:

-   `numpy` para operaciones numéricas.
-   `matplotlib` para la visualización de datos y fronteras de decisión.
-   `scikit-learn` para la generación de datasets de prueba (`make_blobs`).
-   `jupyter` para ejecutar el notebook de pruebas.


## ⚙️ Instalación

1.  **Clona el repositorio:**
    ```bash
    git clone https://github.com/pabcablan/Adaline.git
    cd Adaline
    ```

2.  **Crea un entorno virtual (recomendado):**
    ```bash
    python -m venv venv
    source venv/Scripts/activate
    ```

3.  **Instala las dependencias:**
    ```bash
    pip install numpy matplotlib scikit-learn jupyter
    ```

## 🚀 Ejecución

Para probar el modelo, debes ejecutar el Jupyter Notebook que se encuentra en la carpeta `test/`.

1.  **Inicia Jupyter Notebook:**
    Asegúrate de que tu entorno virtual está activado y ejecuta el siguiente comando desde la raíz del proyecto:
    ```bash
    jupyter notebook
    ```

2.  **Abre y ejecuta el notebook:**
    En la interfaz de Jupyter, navega a la carpeta `test/`, abre el archivo `AdalineTester.ipynb` y ejecuta las celdas en orden. El notebook se encargará de importar el módulo `Adaline` desde `src/`, entrenar el modelo y visualizar los resultados.


## 👥 Autores

Este proyecto fue desarrollado de manera equitativa por:

- [Javier Bolívar García Izquierdo](https://github.com/Javi05x)
- [Pablo Cabeza Lantigua](https://github.com/pabcablan)

La totalidad del trabajo fue realizada de manera simultánea y colaborativa mediante sesiones de programación conjunta a través de Discord.


## 📄 Licencia

Este proyecto está licenciado bajo la licencia MIT. Consulta el archivo `LICENSE` para obtener más detalles.
