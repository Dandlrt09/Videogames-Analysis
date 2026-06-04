import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Configuración de la página
st.set_page_config(
    page_title="Global Sales Predictor - Interactive Market",
    page_icon="📊",
    layout="wide"
)

# Variables globales (reutilizadas en varias partes)
FEATURES = ['Platform', 'Genre', 'Rating', 'Critic_Score', 'User_Score', 'Year_of_Release']
CATEGORICAL_FEATURES = ['Platform', 'Genre', 'Rating']
NUMERIC_FEATURES = ['Critic_Score', 'User_Score', 'Year_of_Release']

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


# ── Entrenar modelos ──
@st.cache_resource
def entrenar_modelos():
    df = cargar_datos()

    # Variables predictoras
    X = df[FEATURES]
    y = df['Global_Sales']

    # Codificación de variables categóricas
    categorical_features = CATEGORICAL_FEATURES
    numeric_features = NUMERIC_FEATURES

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features),
            ('num', 'passthrough', numeric_features)
        ]
    )

    lr_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])

    rf_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    lr_pipeline.fit(X_train, y_train)
    rf_pipeline.fit(X_train, y_train)

    return lr_pipeline, rf_pipeline, X_test, y_test


# ── Cargar todo ──
data = cargar_datos()
lr_model, rf_model, X_test, y_test = entrenar_modelos()

# Predecir con ambos modelos
y_pred_lr = lr_model.predict(X_test)
y_pred_rf = rf_model.predict(X_test)

# Métricas
r2_lr = r2_score(y_test, y_pred_lr)
mae_lr = mean_absolute_error(y_test, y_pred_lr)
rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))

r2_rf = r2_score(y_test, y_pred_rf)
mae_rf = mean_absolute_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))

# Título principal
st.title("📊 Predicción de Ventas Globales - Mercado Interactivo")
st.markdown("---")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Datos", "📈 Evaluación", "🔮 Simulador", "ℹ️ Sobre el proyecto"])

# ═══════════════ Tab 1: Datos ═══════════════
with tab1:
    st.header("📊 Exploración del Dataset")
    st.write("Acá vas a conocer los datos que los modelos usaron para aprender a predecir ventas globales.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Vista de los datos")
        st.caption("Mostrando las primeras 10 filas del dataset:")
        st.dataframe(data.head(10), width='stretch')
        st.success(f"**Total de productos:** {data.shape[0]:,} registros")
        st.info(f"**Variables disponibles:** {data.shape[1]} columnas")

    with col2:
        st.subheader("Estadísticas clave")
        st.caption("Resumen numérico de las variables:")
        st.dataframe(
            data[['Global_Sales', 'NA_Sales', 'EU_Sales', 'JP_Sales', 'Critic_Score', 'User_Score']].describe(),
            width='stretch'
        )

    st.markdown("---")

    st.subheader("📋 ¿Qué variables influyen en las ventas?")
    st.write("Los modelos usan estas variables para predecir las ventas globales:")

    variables_info = pd.DataFrame({
        'Variable': ['Platform', 'Genre', 'Rating', 'Critic_Score', 'User_Score', 'Year_of_Release', 'Global_Sales'],
        '¿Qué es?': [
            'Plataforma de lanzamiento (PS4, XOne, PC, etc.)',
            'Género o categoría (Action, Sports, RPG, etc.)',
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
    ax1.set_ylabel('Cantidad de productos')
    ax1.set_title('Distribución de Ventas Globales')
    plt.tight_layout()
    st.pyplot(fig1)

# ═══════════════ Tab 2: Evaluación ═══════════════
with tab2:
    st.header("📈 ¿Qué tan buenos son los modelos?")
    st.write("Comparamos dos modelos de Machine Learning para predecir ventas globales.")

    st.info("""
    **¿Cómo evaluamos a los modelos?**
    - Los modelos NO vieron estos 3,343 registros durante el entrenamiento
    - Intentan predecir ventas y comparamos con las ventas reales
    """)

    # ── Métricas lado a lado ──
    col_lr, col_rf = st.columns(2)

    with col_lr:
        st.subheader("🔵 Regresión Lineal")
        st.metric("R² Score", f"{r2_lr:.4f}", delta_color="off")
        st.metric("Error Promedio (MAE)", f"{mae_lr:.2f}M", delta_color="off")
        st.metric("Error Cuadrático (RMSE)", f"{rmse_lr:.2f}M", delta_color="off")
        st.caption(f"Explica el {r2_lr:.1%} de la varianza.")

    with col_rf:
        st.subheader("🟢 Random Forest (100 árboles)")
        st.metric("R² Score", f"{r2_rf:.4f}", delta_color="off")
        st.metric("Error Promedio (MAE)", f"{mae_rf:.2f}M", delta_color="off")
        st.metric("Error Cuadrático (RMSE)", f"{rmse_rf:.2f}M", delta_color="off")
        st.caption(f"Explica el {r2_rf:.1%} de la varianza.")

    # Comparación
    mejor = "Random Forest" if r2_rf > r2_lr else "Regresión Lineal"
    st.success(f"🏆 **{mejor} tiene mejor rendimiento en este dataset.**")
    # Mostrar diferencia
    diff_r2 = abs(r2_rf - r2_lr)
    st.info(f"📊 **Diferencia de R² entre modelos:** {diff_r2:.4f}")

    # ── Features más importantes (Random Forest) ──
    with st.expander("🌲 ¿Qué variables son más importantes para Random Forest?"):
        # Obtener nombres de las features transformadas
        encoder = rf_model.named_steps['preprocessor'].named_transformers_['cat']
        feature_names_cat = encoder.get_feature_names_out(CATEGORICAL_FEATURES)
        all_feature_names = np.concatenate([feature_names_cat, NUMERIC_FEATURES])
        importances = rf_model.named_steps['regressor'].feature_importances_

        # Top 10
        idx_top = np.argsort(importances)[-10:][::-1]
        top_nombres = [all_feature_names[i] for i in idx_top]
        top_importancias = [importances[i] for i in idx_top]

        fig_imp, ax_imp = plt.subplots(figsize=(8, 4))
        ax_imp.barh(range(len(top_nombres)), top_importancias, color='forestgreen')
        ax_imp.set_yticks(range(len(top_nombres)))
        ax_imp.set_yticklabels(top_nombres)
        ax_imp.set_xlabel('Importancia relativa')
        ax_imp.set_title('Top 10 variables más importantes (Random Forest)')
        plt.tight_layout()
        st.pyplot(fig_imp)

        st.caption("Las variables categóricas se expanden en múltiples columnas (una por categoría).")

    st.markdown("---")

    # ── Visualización ──
    st.subheader("📊 Comparación visual")
    st.write("**Gráfico 1: Real vs Predicho**")

    col1, col2 = st.columns(2)

    with col1:
        fig_lr, ax_lr = plt.subplots(figsize=(6, 4))
        ax_lr.scatter(y_test, y_pred_lr, alpha=0.3, s=5, color='dodgerblue')
        ax_lr.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Predicción perfecta')
        ax_lr.set_xlabel('Ventas Reales (M)')
        ax_lr.set_ylabel('Ventas Predichas (M)')
        ax_lr.set_title(f'🔵 Regresión Lineal (R²={r2_lr:.3f})')
        ax_lr.legend()
        ax_lr.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig_lr)
        st.caption("Los puntos cerca de la línea roja = predicción cercana a la realidad.")

    with col2:
        fig_rf, ax_rf = plt.subplots(figsize=(6, 4))
        ax_rf.scatter(y_test, y_pred_rf, alpha=0.3, s=5, color='forestgreen')
        ax_rf.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Predicción perfecta')
        ax_rf.set_xlabel('Ventas Reales (M)')
        ax_rf.set_ylabel('Ventas Predichas (M)')
        ax_rf.set_title(f'🟢 Random Forest (R²={r2_rf:.3f})')
        ax_rf.legend()
        ax_rf.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig_rf)
        st.caption("Los puntos cerca de la línea roja = predicción cercana a la realidad.")

    st.markdown("---")
    st.write("**Gráfico 2: Distribución de errores**")

    col1, col2 = st.columns(2)

    with col1:
        errors_lr = y_test.values - y_pred_lr
        fig_e1, ax_e1 = plt.subplots(figsize=(6, 3))
        ax_e1.hist(errors_lr, bins=50, edgecolor='black', color='skyblue', alpha=0.7)
        ax_e1.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Error cero')
        ax_e1.set_xlabel('Error (M)')
        ax_e1.set_ylabel('Cantidad de productos')
        ax_e1.set_title('🔵 Regresión Lineal')
        ax_e1.legend()
        ax_e1.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig_e1)
        st.caption(f"Error promedio: ±{mae_lr:.2f}M")

    with col2:
        errors_rf = y_test.values - y_pred_rf
        fig_e2, ax_e2 = plt.subplots(figsize=(6, 3))
        ax_e2.hist(errors_rf, bins=50, edgecolor='black', color='lightgreen', alpha=0.7)
        ax_e2.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Error cero')
        ax_e2.set_xlabel('Error (M)')
        ax_e2.set_ylabel('Cantidad de productos')
        ax_e2.set_title('🟢 Random Forest')
        ax_e2.legend()
        ax_e2.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig_e2)
        st.caption(f"Error promedio: ±{mae_rf:.2f}M")

    st.warning(f"""
    ⚠️ **Nota sobre el rendimiento:** 
    - Regresión Lineal: R² de {r2_lr:.3f}. Simple, interpretable, pero asume relaciones lineales.
    - Random Forest: R² de {r2_rf:.3f}. Captura relaciones no lineales, pero es más complejo.
    - Las ventas en el mercado interactivo dependen de muchos factores no capturados (marketing, secuelas, etc.).
    """)

# ═══════════════ Tab 3: Simulador ═══════════════
with tab3:
    st.header("🔮 Simulador de Ventas")
    st.write("Ajustá los parámetros del producto para predecir sus ventas globales:")

    modelo_elegido = st.selectbox(
        "Seleccioná el modelo para predecir:",
        ["Random Forest", "Regresión Lineal"],
        help="Random Forest suele dar mejores predicciones para este dataset"
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Características del producto")
        platform = st.selectbox("Plataforma", sorted(data['Platform'].unique()))
        genre = st.selectbox("Género", sorted(data['Genre'].unique()))
        rating = st.selectbox("Clasificación ESRB", sorted(data['Rating'].unique()))

    with col2:
        st.subheader("Puntuaciones y fecha")
        critic_score = st.slider("Puntuación de Críticos (0-100)", 0, 100, 70)
        user_score = st.slider("Puntuación de Usuarios (0-10)", 0.0, 10.0, 7.0, 0.1)
        year = st.slider("Año de Lanzamiento", 1980, 2020, 2010)

    if st.button("🔮 Predecir Ventas", type="primary", width='stretch'):
        input_data = pd.DataFrame({
            'Platform': [platform],
            'Genre': [genre],
            'Rating': [rating],
            'Critic_Score': [critic_score],
            'User_Score': [user_score],
            'Year_of_Release': [year]
        })

        modelo = rf_model if modelo_elegido == "Random Forest" else lr_model
        prediction = modelo.predict(input_data)[0]

        st.success(f"**Ventas globales estimadas: {prediction:.2f} millones**")
        st.info(f"💡 *Predicción generada con **{modelo_elegido}**. Valor referencial, no una garantía.*")

        st.write(f"Eso equivale a **${prediction * 1000000:,.0f}** en ventas totales (asumiendo un precio promedio de $60).")

        # Mostrar distribución similar
        st.caption("💡 *El modelo se entrenó con datos de productos lanzados entre 1980 y 2016. Predicciones fuera de ese rango son menos confiables.*")

# ═══════════════ Tab 4: Sobre el proyecto ═══════════════
with tab4:
    st.header("ℹ️ Sobre este proyecto")

    st.markdown(f"""
    ### 🎯 Objetivo
    Este proyecto analiza el comportamiento de las ventas globales de productos de
    entretenimiento interactivo utilizando técnicas de análisis de datos y machine learning.
    El dataset contiene información sobre más de **16,000 lanzamientos** entre 1985 y 2016.

    ### 🛠️ Herramientas utilizadas
    - **Python** y **Pandas** para manipulación de datos
    - **Scikit-learn** para el modelado (Regresión Lineal + Random Forest)
    - **Matplotlib** para visualizaciones
    - **Streamlit** para el despliegue interactivo

    ### 📊 Dataset
    El dataset incluye más de 16,000 registros de lanzamientos con:
    - Variables: Plataforma, Género, Año, Críticas, Ventas por región
    - **Variable objetivo**: Global_Sales (ventas globales en millones)

    ### 🏆 Rendimiento de los modelos
    | Modelo | R² Score | MAE |
    |--------|:--------:|:---:|
    | Regresión Lineal | {r2_lr:.4f} | {mae_lr:.2f}M |
    | Random Forest | {r2_rf:.4f} | {mae_rf:.2f}M |

    - Regresión Lineal: simple e interpretable, pero asume relaciones lineales.
    - Random Forest: captura relaciones no lineales e interacciones entre variables.

    ### 📁 Repositorio
    Podés ver el notebook original con el análisis completo en:
    [DataScience_Proyects - Proyecto 2](https://github.com/Dandlrt09/DataScience_Proyects/tree/main/Proyecto%202)

    **Nota**: Este análisis es educativo. Los hallazgos deben considerarse como
    insights exploratorios, no como consejos de inversión.
    """)

# Footer
st.markdown("---")
st.caption("📊 Global Sales Predictor - Interactive Market | Desarrollado con Streamlit y Scikit-learn")
