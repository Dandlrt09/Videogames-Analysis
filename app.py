import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Configuración de la página
st.set_page_config(
    page_title="Video Games Sales Predictor",
    page_icon="🎮",
    layout="wide"
)

# ── Cargar datos ──
@st.cache_data
def cargar_datos():
    df = pd.read_csv('video_games.csv')
    # Limpiar datos
    df = df.dropna(subset=['Global_Sales'])
    df['Critic_Score'] = df['Critic_Score'].fillna(df['Critic_Score'].median())
    df['User_Score'] = df['User_Score'].replace('tbd', np.nan)
    df['User_Score'] = pd.to_numeric(df['User_Score'], errors='coerce')
    df['User_Score'] = df['User_Score'].fillna(df['User_Score'].median())
    df['Year_of_Release'] = df['Year_of_Release'].fillna(df['Year_of_Release'].mode()[0])
    df['Rating'] = df['Rating'].fillna('E')
    df['Genre'] = df['Genre'].fillna('Unknown')
    df['Platform'] = df['Platform'].fillna('Unknown')
    return df

# ── Entrenar modelo ──
@st.cache_resource
def entrenar_modelo():
    df = cargar_datos()
    
    # Variables predictoras
    features = ['Platform', 'Genre', 'Rating', 'Critic_Score', 'User_Score', 'Year_of_Release']
    X = df[features]
    y = df['Global_Sales']
    
    # Codificación de variables categóricas
    categorical_features = ['Platform', 'Genre', 'Rating']
    numeric_features = ['Critic_Score', 'User_Score', 'Year_of_Release']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features),
            ('num', 'passthrough', numeric_features)
        ]
    )
    
    modelo = LinearRegression()
    pipe = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', modelo)])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipe.fit(X_train, y_train)
    
    return pipe, X_test, y_test

# Cargar y entrenar
data = cargar_datos()
model, X_test, y_test = entrenar_modelo()

# Título principal
st.title("🎮 Predicción de Ventas de Videojuegos")
st.markdown("---")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Datos", "📈 Evaluación", "🎮 Simulador", "ℹ️ Sobre el proyecto"])

# Tab 1: Datos
with tab1:
    st.header("📊 Exploración del Dataset")
    st.write("Acá vas a conocer los datos que el modelo usó para aprender a predecir ventas globales.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Vista de los datos")
        st.caption("Mostrando las primeras 10 filas del dataset:")
        st.dataframe(data.head(10), width='stretch')
        st.success(f"**Total de juegos:** {data.shape[0]:,} registros")
        st.info(f"**Variables disponibles:** {data.shape[1]} columnas")
    
    with col2:
        st.subheader("Estadísticas clave")
        st.caption("Resumen numérico de las variables:")
        st.dataframe(data[['Global_Sales', 'NA_Sales', 'EU_Sales', 'JP_Sales', 'Critic_Score', 'User_Score']].describe(), width='stretch')
    
    st.markdown("---")
    
    st.subheader("📋 ¿Qué variables influyen en las ventas?")
    st.write("El modelo usa estas variables para predecir las ventas globales:")
    
    variables_info = pd.DataFrame({
        'Variable': ['Platform', 'Genre', 'Rating', 'Critic_Score', 'User_Score', 'Year_of_Release', 'Global_Sales'],
        '¿Qué es?': [
            'Plataforma de lanzamiento (PS4, XOne, PC, etc.)',
            'Género del juego (Action, Sports, RPG, etc.)',
            'Clasificación ESRB (E, T, M, etc.)',
            'Puntuación de críticos (0-100)',
            'Puntuación de usuarios (0-10)',
            'Año de lanzamiento',
            'Ventas globales en millones (variable objetivo)'
        ],
        'Tipo': ['Categórica', 'Categórica', 'Categórica', 'Numérico', 'Numérico', 'Numérico', 'Numérico']
    })
    
    st.dataframe(variables_info, width='stretch', hide_index=True)
    
    st.subheader("Distribución de ventas globales")
    fig1, ax1 = plt.subplots(figsize=(8, 4))
    ax1.hist(data['Global_Sales'], bins=50, edgecolor='black', alpha=0.7, color='skyblue')
    ax1.set_xlabel('Ventas Globales (millones)')
    ax1.set_ylabel('Cantidad de juegos')
    ax1.set_title('Distribución de Ventas Globales')
    plt.tight_layout()
    st.pyplot(fig1)

# Tab 2: Evaluación
with tab2:
    st.header("📈 ¿Qué tan bueno es el modelo?")
    st.write("Evaluamos la capacidad predictiva del modelo de Regresión Lineal.")
    
    y_pred = model.predict(X_test)
    
    st.subheader("🎯 Métricas de rendimiento")
    st.info("""
    **¿Cómo evaluamos al modelo?**
    - El modelo NO vio estos 3,343 juegos durante el entrenamiento
    - Intenta predecir ventas y comparamos con las ventas reales
    """)
    
    col1, col2, col3 = st.columns(3)
    
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    
    with col1:
        st.metric("R² Score", f"{r2:.4f}")
        st.caption("Explica el 39.8% de la varianza. No es perfecto pero útil.")
    
    with col2:
        st.metric("Error Promedio (MAE)", f"{mae:.2f}M")
        st.caption("En promedio se equivoca por ~0.5 millones de ventas.")
    
    with col3:
        st.metric("Error Cuadrático (RMSE)", f"{rmse:.2f}M")
        st.caption("Castiga errores grandes. Más sensible que MAE.")
    
    st.warning("""
    ⚠️ **Nota sobre el rendimiento:** Un R² de 0.398 es razonable para un modelo simple.
    Las ventas de videojuegos dependen de muchos factores no capturados (marketing, secuelas, etc.).
    """)
    
    st.markdown("---")
    
    st.subheader("📊 Visualización del rendimiento")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Gráfico 1: Real vs Predicho**")
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        ax2.scatter(y_test, y_pred, alpha=0.3, s=5, color='dodgerblue')
        ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Predicción perfecta')
        ax2.set_xlabel('Ventas Reales (M)')
        ax2.set_ylabel('Ventas Predichas (M)')
        ax2.set_title('¿Acierta el modelo?')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig2)
        
        st.caption("Los puntos cerca de la línea roja = predicción cercana a la realidad.")
    
    with col2:
        st.write("**Gráfico 2: Distribución de errores**")
        errors = y_test.values - y_pred
        fig3, ax3 = plt.subplots(figsize=(6, 4))
        ax3.hist(errors, bins=50, edgecolor='black', color='skyblue', alpha=0.7)
        ax3.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Error cero')
        ax3.set_xlabel('Error (M)')
        ax3.set_ylabel('Cantidad de juegos')
        ax3.set_title('¿Cuánto se equivocó?')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig3)
        
        st.caption("La mayoría de los errores cerca de 0 = modelo confiable.")

# Tab 3: Simulador
with tab3:
    st.header("🎮 Simulador de Ventas")
    st.write("Ajustá los parámetros del juego para predecir sus ventas globales:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Características del juego")
        platform = st.selectbox("Plataforma", sorted(data['Platform'].unique()))
        genre = st.selectbox("Género", sorted(data['Genre'].unique()))
        rating = st.selectbox("Clasificación ESRB", sorted(data['Rating'].unique()))
        
    with col2:
        st.subheader("Puntuaciones y fecha")
        critic_score = st.slider("Puntuación de Críticos (0-100)", 0, 100, 70)
        user_score = st.slider("Puntuación de Usuarios (0-10)", 0.0, 10.0, 7.0, 0.1)
        year = st.slider("Año de Lanzamiento", 1980, 2020, 2010)
    
    if st.button("🎮 Predecir Ventas", type="primary", width='stretch'):
        input_data = pd.DataFrame({
            'Platform': [platform],
            'Genre': [genre],
            'Rating': [rating],
            'Critic_Score': [critic_score],
            'User_Score': [user_score],
            'Year_of_Release': [year]
        })
        
        prediction = model.predict(input_data)[0]
        
        st.success(f"**Ventas globales estimadas: {prediction:.2f} millones**")
        st.info("💡 *Este es un valor referencial basado en el modelo de Machine Learning entrenado.*")
        
        # Mostrar cuánto representa
        st.write(f"Eso equivale a **${prediction * 1000000:,.0f}** en ventas totales (asumiendo un precio promedio).")

# Tab 4: Sobre el proyecto
with tab4:
    st.header("ℹ️ Sobre este proyecto")
    
    st.markdown("""
    ### 🎯 Objetivo
    Este proyecto analiza el comportamiento de las ventas de videojuegos a nivel global
    utilizando técnicas de análisis de datos y machine learning. El dataset contiene
    información sobre más de **16,000 juegos** lanzados entre 2000 y 2016.
    
    ### 🛠️ Herramientas utilizadas
    - **Python** y **Pandas** para manipulación de datos
    - **Scikit-learn** para el modelado (Regresión Lineal)
    - **Matplotlib** para visualizaciones
    - **Streamlit** para el despliegue interactivo
    
    ### 📊 Dataset
    El dataset "Video Games Sales" incluye:
    - **16,719 registros** de juegos
    - Variables: Plataforma, Género, Año, Críticas, Ventas por región
    - **Variable objetivo**: Global_Sales (ventas globales en millones)
    
    ### 🏆 Rendimiento del modelo
    - **R² Score**: 0.398 (explica ~40% de la varianza)
    - **Mean Absolute Error**: ~0.5 millones de ventas
    - **Algoritmo**: Regresión Lineal (simple pero interpretable)
    
    ### 📁 Repositorio
    Podés ver el notebook original en:
    [DataScience_Proyects - Proyecto 2](https://github.com/Dandlrt09/DataScience_Proyects/tree/main/Proyecto%202)
    
    **Nota**: Este análisis es educativo. Los hallazgos deben considerarse como
    insights exploratorios, no como consejos de inversión.
    """)

# Footer
st.markdown("---")
st.caption("🎮 Video Games Sales Predictor | Desarrollado con Streamlit y Scikit-learn")
