import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import hypergeom

# Configuración de la página
st.set_page_config(page_title="Calculadora Estadística", page_icon="📊", layout="wide")

st.title("Calculadora de Distribuciones Discretas")
st.markdown("Herramienta interactiva para calcular probabilidades y visualizar el comportamiento de variables aleatorias discretas.")

# barra lateral
st.sidebar.header("Configuración de Parámetros")
distribucion = st.sidebar.selectbox(
    "Selecciona la distribución", 
    ["Hipergeométrica", "Binomial (Próximamente)", "Poisson (Próximamente)"]
)

if distribucion == "Hipergeométrica":
    st.header("Distribución Hipergeométrica")
    st.markdown("""
    Se utiliza para calcular la probabilidad de obtener $x$ éxitos en una muestra de tamaño $n$, 
    extraída de una población de tamaño $N$ que contiene $K$ éxitos en total, **sin reemplazo**.
    """)
    
    col1, col2 = st.columns([1, 1.5])
    
    with col1:
        st.subheader("Ingresa los datos")
        N = st.number_input("Tamaño de la población (N)", min_value=1, value=None, step=1, placeholder="Ej: 50")
        K = st.number_input("Éxitos en la población (K)", min_value=0, value=None, step=1, placeholder="Ej: 10")
        n = st.number_input("Tamaño de la muestra (n)", min_value=1, value=None, step=1, placeholder="Ej: 5")
        x = st.number_input("Éxitos en la muestra (x)", min_value=0, value=None, step=1, placeholder="Ej: 2")

    with col2:
        st.subheader("Resultados y Procedimiento")
        
        # Verificar que el usuario ya metió todos los valores
        if N is not None and K is not None and n is not None and x is not None:
            # Castear a enteros para seguridad en los cálculos y visualización
            N, K, n, x = int(N), int(K), int(n), int(x)
            
            # Validaciones matemáticas
            errores = False
            if K > N:
                st.error("Error: Los éxitos en la población (K) no pueden ser mayores a la población total (N).")
                errores = True
            if n > N:
                st.error("Error: El tamaño de la muestra (n) no puede ser mayor a la población total (N).")
                errores = True
            if x > min(n, K):
                st.error(f"Error: Los éxitos en la muestra (x) no pueden superar al tamaño de la muestra ({n}) ni a los éxitos totales ({K}).")
                errores = True
                
            if not errores:
                # Cálculos
                rv = hypergeom(N, K, n)
                prob_exacta = rv.pmf(x)
                prob_acumulada_menor = rv.cdf(x)
                prob_acumulada_mayor = rv.sf(x - 1)  # P(X >= x) es equivalente a 1 - P(X <= x-1)
                esperanza = rv.mean()
                varianza = rv.var()

                # Mostrar fórmulas 
                st.markdown("### Desarrollo didáctico")
                st.markdown("**1. Fórmula de la probabilidad exacta:**")
                st.latex(r"P(X = x) = \frac{\binom{K}{x} \binom{N-K}{n-x}}{\binom{N}{n}}")
                st.latex(rf"P(X = {x}) = \frac{{\binom{{{K}}}{{{x}}} \binom{{{N}-{K}}}{{{n}-{x}}}}}{{\binom{{{N}}}{{{n}}}}} = {prob_exacta:.4f}")
                
                st.markdown("**2. Esperanza matemática o media:**")
                st.latex(r"\mu = n \cdot \frac{K}{N}")
                st.latex(rf"\mu = {n} \cdot \frac{{{K}}}{{{N}}} = {esperanza:.4f}")
                
                st.divider()
                
                # Mostrar resultados finales
                st.markdown("### Resumen de Probabilidades")
                st.info(f"🎯 **Probabilidad Exacta $P(X = {x})$:** {prob_exacta:.4f} ({(prob_exacta*100):.2f}%)")
                st.success(f"📉 **Acumulada hacia abajo (máximo {x} éxitos) $P(X \le {x})$:** {prob_acumulada_menor:.4f} ({(prob_acumulada_menor*100):.2f}%)")
                st.warning(f"📈 **Acumulada hacia arriba (al menos {x} éxitos) $P(X \ge {x})$:** {prob_acumulada_mayor:.4f} ({(prob_acumulada_mayor*100):.2f}%)")
                
                col_res1, col_res2 = st.columns(2)
                col_res1.metric(label="Esperanza matemática o media (μ)", value=f"{esperanza:.4f}")
                col_res2.metric(label="Varianza (σ²)", value=f"{varianza:.4f}")

    # Gráfica y tabla
    if N is not None and K is not None and n is not None and x is not None and not errores:
        st.divider()
        st.subheader("Visualización de la Función de Masa de Probabilidad")
        
        st.markdown("""
        💡 **¿Cómo interpretar esta gráfica?**
        * **Eje Horizontal (X):** Muestra todos los posibles números de éxitos que podrías obtener en tu muestra.
        * **Eje Vertical (Y):** Indica qué tan probable es que ocurra exactamente ese número de éxitos.
        * **Barra Roja:** Resalta el valor $x$ que calculaste. Las barras más altas te indican cuáles son los resultados más comunes o esperados.
        """)
        
        max_x = min(n, K)
        eje_x = np.arange(0, max_x + 1)
        pmf_values = rv.pmf(eje_x)

        fig, ax = plt.subplots(figsize=(10, 4))
        bars = ax.bar(eje_x, pmf_values, color='#3498db', edgecolor='black', alpha=0.7)
        
        if x in eje_x:
            bars[x].set_color('#e74c3c')
            bars[x].set_alpha(1.0)
        
        ax.set_xlabel('Número de éxitos en la muestra (x)')
        ax.set_ylabel('Probabilidad P(X=x)')
        ax.set_xticks(eje_x)
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        st.pyplot(fig)
        plt.close(fig)
        
        # Tabla de distribución
        st.markdown("### Tabla de Distribución de Probabilidades")
        st.markdown("Valores exactos para cada posible número de éxitos en la muestra:")
        
        # Crear un DataFrame de pandas para la tabla
        df_distribucion = pd.DataFrame({
            "Éxitos (x)": eje_x,
            "Probabilidad Exacta P(X=x)": [f"{p:.6f}" for p in pmf_values], # Formato a 6 decimales
            "Probabilidad (%)": [f"{(p * 100):.4f}%" for p in pmf_values]
        })
        
        # Mostrar la tabla en streamlit
        st.dataframe(df_distribucion, hide_index=True, use_container_width=True)

    else:
        st.info("👈 Ingresa los 4 parámetros en la barra lateral para generar los cálculos y la gráfica.")

else:
    st.info("Esta distribución está en desarrollo. Por favor, selecciona la Hipergeométrica.")