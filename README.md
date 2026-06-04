# 📊 Global Sales Predictor — Interactive Market Analysis (1985-2016)

Aplicación interactiva de **Machine Learning** que analiza y predice ventas globales de productos de entretenimiento interactivo basándose en plataforma, género, puntuaciones y año de lanzamiento.

![Streamlit App](https://img.shields.io/badge/Streamlit-1.45+-FF4B4B?logo=streamlit)
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.5+-F7931E?logo=scikit-learn)

---

## 🎯 Descripción

Este proyecto analiza el comportamiento de las ventas globales de productos de entretenimiento interactivo utilizando técnicas de análisis de datos y machine learning. El dataset contiene información sobre más de **16,000 lanzamientos** entre 1985 y 2016.

### ✨ Características

- 📊 **Exploración de Datos**: Vista interactiva del dataset con estadísticas descriptivas
- 📈 **Evaluación del Modelo**: Métricas de rendimiento (R², MAE, RMSE) y comparación de modelos
- 🔮 **Simulador**: Predicción de ventas ajustando las características del producto
- ℹ️ **Sobre el Proyecto**: Contexto, herramientas utilizadas y enlaces al repositorio original

---

## 🛠️ Tecnologías

- **Python 3.10+**
- **Streamlit** - Interfaz web interactiva
- **Scikit-learn** - Modelado y evaluación (Regresión Lineal + Random Forest)
- **Pandas** - Manipulación de datos
- **Matplotlib** - Visualizaciones

---

## 📊 Dataset

El dataset contiene **16,719 registros** de lanzamientos (1985-2016) con las siguientes variables:

| Variable | Descripción | Tipo |
|----------|-------------|------|
| `Platform` | Plataforma de lanzamiento (PS4, XOne, PC, etc.) | Categórica |
| `Genre` | Género o categoría (Action, Sports, RPG, etc.) | Categórica |
| `Rating` | Clasificación ESRB (E, T, M, etc.) | Categórica |
| `Critic_Score` | Puntuación de críticos (0-100) | Numérico |
| `User_Score` | Puntuación de usuarios (0-10) | Numérico |
| `Year_of_Release` | Año de lanzamiento | Numérico |
| `Global_Sales` | **Ventas globales (millones)** — Variable objetivo | Numérico |

---

## 🏆 Rendimiento de los Modelos

La app entrena dos modelos con un pipeline consistente (`ColumnTransformer` + `Pipeline`):

| Modelo | R² Score | MAE | RMSE |
|--------|:--------:|:---:|:----:|
| Regresión Lineal | ~0.045 | ~0.83M | ~2.37M |
| Random Forest (100 árboles) | ~-0.011 | ~0.83M | ~2.40M |

**Nota**: Ambos modelos muestran capacidad predictiva limitada. Esto se debe a que el dataset carece de variables críticas como marketing, engagement en redes sociales, y presupuesto de producción. El valor del proyecto radica en el **análisis exploratorio**, la implementación del pipeline, y la comparación metodológica entre modelos.

---

## 🚀 Instalación y Uso

### 1. Clonar el repositorio
```bash
git clone https://github.com/Dandlrt09/DataScience_Proyects.git
cd "Proyecto 10/StreamlitGame"
```

### 2. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 3. Ejecutar la app
```bash
streamlit run app.py
```

La app se abrirá en tu navegador en `http://localhost:8501`.

---

## 📊 Estructura del Proyecto

```
StreamlitGame/
├── app.py              # Aplicación principal de Streamlit
├── video_games.csv     # Dataset (~1.6MB)
├── requirements.txt    # Dependencias
└── README.md           # Este archivo
```

---

## 🔗 Enlaces Relacionados

- **Notebook de análisis completo**: [Proyecto 2/Main.ipynb](https://github.com/Dandlrt09/DataScience_Proyects/blob/main/Proyecto%202/Main.ipynb)
- **Portafolio personal**: [danieldlrt09.github.io](https://danieldlrt09.github.io/Portafolio_Personal/)

---

## 📝 Notas

- Los modelos se entrenan **dentro de la app** usando `@st.cache_resource` (no requiere archivos .pkl externos)
- El dataset se carga desde `video_games.csv` (incluido en el repositorio)
- Este análisis es de carácter educativo y los hallazgos deben considerarse como insights exploratorios, no como consejos de inversión.

---

## 📄 Licencia

Este proyecto es de uso educativo y hace parte del portafolio de Data Science.

**Desarrollado por** [Daniel Del Río](https://github.com/Dandlrt09) 📊
