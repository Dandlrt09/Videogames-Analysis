# 🎮 Video Games Sales Predictor

Una aplicación interactiva de **Machine Learning** que predice las ventas globales de videojuegos basándose en plataforma, género, puntuaciones y año de lanzamiento.

![Streamlit App](https://img.shields.io/badge/Streamlit-1.45+-FF4B4B?logo=streamlit)
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.5+-F7931E?logo=scikit-learn)

---

## 🎯 Descripción

Este proyecto analiza el comportamiento de las ventas de videojuegos a nivel global utilizando técnicas de análisis de datos y machine learning. El dataset contiene información sobre más de **16,000 juegos** lanzados entre 2000 y 2016.

### ✨ Características

- 📊 **Exploración de Datos**: Vista interactiva del dataset con estadísticas descriptivas
- 📈 **Evaluación del Modelo**: Métricas de rendimiento (R², MAE, RMSE) y visualizaciones
- 🎮 **Simulador**: Predicción de ventas ajustando las características del juego
- ℹ️ **Sobre el Proyecto**: Contexto, herramientas utilizadas y enlaces al repositorio original

---

## 🛠️ Tecnologías

- **Python 3.10+**
- **Streamlit** - Interfaz web interactiva
- **Scikit-learn** - Modelado y evaluación (Regresión Lineal)
- **Pandas** - Manipulación de datos
- **Matplotlib** - Visualizaciones

---

## 📊 Dataset

El dataset "Video Games Sales as at 22 Dec 2016" contiene **16,719 registros** con las siguientes variables:

| Variable | Descripción | Tipo |
|----------|-------------|------|
| `Platform` | Plataforma de lanzamiento (PS4, XOne, PC, etc.) | Categórica |
| `Genre` | Género del juego (Action, Sports, RPG, etc.) | Categórica |
| `Rating` | Clasificación ESRB (E, T, M, etc.) | Categórica |
| `Critic_Score` | Puntuación de críticos (0-100) | Numérico |
| `User_Score` | Puntuación de usuarios (0-10) | Numérico |
| `Year_of_Release` | Año de lanzamiento | Numérico |
| `Global_Sales` | **Ventas globales (millones)** - Variable objetivo | Numérico |

---

## 🏆 Rendimiento del Modelo

El modelo utilizado es **Regresión Lineal** (simple pero interpretable):

| Métrica | Valor | Interpretación |
|---------|-------|----------------|
| **R² Score** | 0.398 | Explica ~40% de la varianza en ventas |
| **MAE** | ~0.5M | Error promedio de 500,000 ventas |
| **RMSE** | ~1.2M | Error sensible a desviaciones grandes |

**Nota**: Un R² de 0.398 es razonable para un modelo simple. Las ventas de videojuegos dependen de muchos factores no capturados (marketing, secuelas, hype, etc.).

---

## 🚀 Instalación y Uso

### 1. Clonar el repositorio
```bash
git clone https://github.com/Dandlrt09/Videogames-Analysis.git
cd Videogames-Analysis
```

### 2. Instalar dependencias
```bash
pip install -r requirements.txt
```
*(Si no tenés el archivo, instala manualmente: `pip install streamlit pandas scikit-learn matplotlib numpy`)*

### 3. Ejecutar la app
```bash
streamlit run app.py
```

La app se abrirá en tu navegador en `http://localhost:8501`.

---

## 📊 Estructura del Proyecto

```
Videogames-Analysis/
├── app.py              # Aplicación principal de Streamlit
├── video_games.csv    # Dataset (1.6MB)
├── requirements.txt    # Dependencias (solo 4 librerías)
└── README.md          # Este archivo
```

---

## 🔗 Enlaces Relacionados

- **Repositorio original del proyecto**: [DataScience_Proyects - Proyecto 2](https://github.com/Dandlrt09/DataScience_Proyects/tree/main/Proyecto%202)
- **Notebook de entrenamiento**: [Main.ipynb](https://github.com/Dandlrt09/DataScience_Proyects/blob/main/Proyecto%202/Main.ipynb)
- **Portafolio personal**: [danieldlrt09.github.io/Portafolio_Personal](https://danieldlrt09.github.io/Portafolio_Personal/)

---

## 📝 Notas

- El modelo se entrena **adentro de la app** usando `@st.cache_resource` (no requiere archivos .pkl)
- El dataset se carga desde `video_games.csv` (disponible en el repo)
- Este análisis es de carácter educativo y los hallazgos deben considerarse como insights exploratorios, no como consejos de inversión.

---

## 📄 Licencia

Este proyecto es de uso educativo y hace parte del portafolio de Data Science.

**Desarrollado por** [Daniel Del Río](https://github.com/Dandlrt09) 🎮
