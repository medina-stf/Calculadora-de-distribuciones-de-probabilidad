import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from scipy.stats import hypergeom

# 1. Configuración principal
st.set_page_config(page_title="Calculadora de Distribución Hipergeométrica", page_icon="", layout="wide")

st.title("Calculadora Interactiva de Probabilidad")
st.markdown("Explora el comportamiento de la distribución hipergeométrica con interactividad en tiempo real y simulaciones computacionales.")

# 2. Barra lateral con entradas numéricas libres
st.sidebar.header("Panel de Control")
st.sidebar.markdown("Ingresa los valores. El sistema validará la lógica matemática en tiempo real.")

distribucion = st.sidebar.selectbox("Selecciona la distribución", ["Hipergeométrica"])

# Entradas por teclado
N = st.sidebar.number_input("Población Total (N)", value=50, step=1)
K = st.sidebar.number_input("Éxitos en la Población (K)", value=10, step=1)
n = st.sidebar.number_input("Tamaño de la Muestra (n)", value=5, step=1)
x = st.sidebar.number_input("Éxitos deseados en la muestra (x)", value=2, step=1)

# Radiografía visual de la población
if N > 0 and 0 <= K <= N:
    fig_pie = px.pie(
        values=[K, N - K], 
        names=["Éxitos (Defectuosos/Objetivo)", "Fracasos (Resto)"], 
        title="Composición de la Población",
        color_discrete_sequence=['#e74c3c', '#3498db'],
        hole=0.4
    )
    fig_pie.update_layout(margin=dict(t=30, b=0, l=0, r=0), showlegend=False)
    st.sidebar.plotly_chart(fig_pie, use_container_width=True)

# 3. Escudo de validación
errores = False

if N < 1:
    st.error("⚠️ **Error Lógico:** La Población Total (N) debe ser de al menos 1 artículo.")
    errores = True
elif K < 0 or K > N:
    st.error(f"⚠️ **Error Lógico:** Los éxitos totales (K) no pueden ser negativos ni mayores a tu población total ({N}).")
    errores = True
elif n < 1 or n > N:
    st.error(f"⚠️ **Error Lógico:** El tamaño de la muestra (n) debe ser al menos 1 y no puede ser más grande que la caja entera ({N}).")
    errores = True
else:
    max_x_posible = min(n, K)
    if x < 0 or x > max_x_posible:
        st.error(f"⚠️ **Error Lógico:** Pides {x} éxitos, pero matemáticamente solo puedes sacar entre 0 y {max_x_posible} éxitos con tu configuración actual.")
        errores = True

# Si todo es correcto, ejecutamos los cálculos
if not errores:
    rv = hypergeom(N, K, n)
    prob_exacta = rv.pmf(x)
    prob_acum_menor = rv.cdf(x)
    prob_acum_mayor = rv.sf(x - 1)
    esperanza = rv.mean()
    varianza = rv.var()

    limite_grafica = min(max_x_posible, 1000) 
    eje_x = np.arange(0, limite_grafica + 1)
    pmf_values = rv.pmf(eje_x)

    st.success(f"""
    **Traducción del problema:** Imagina que tienes un lote con **{N:,} artículos**, de los cuales **{K:,} son tu objetivo**. 
    Si sacas **{n:,} artículos** al azar de un solo golpe, la probabilidad matemática de sacar 
    exactamente **{x:,} artículos objetivo** es del **{(prob_exacta*100):.4f}%**.
    """)

    # 4. Pestañas de Navegación
    tab1, tab2, tab3, tab4 = st.tabs(["Análisis e Intervalos", "Gráfica Interactiva", "Simulador Monte Carlo", "Fórmulas e Interpretación"])
    
    with tab1:
        st.markdown("### Resumen de Probabilidades")
        col1, col2, col3 = st.columns(3)
        col1.metric("Exactamente x", f"{(prob_exacta*100):.4f}%", f"P(X = {x})")
        col2.metric("Máximo x", f"{(prob_acum_menor*100):.4f}%", f"P(X ≤ {x})")
        col3.metric("Al menos x", f"{(prob_acum_mayor*100):.4f}%", f"P(X ≥ {x})")
        
        st.divider()
        
        st.markdown("### Calculadora de Rangos (Probabilidad por Intervalo)")
        col_int1, col_int2 = st.columns(2)
        min_int = col_int1.number_input("Mínimo esperado (a)", value=max(0, x-1), step=1)
        max_int = col_int2.number_input("Máximo esperado (b)", value=min(max_x_posible, x+1), step=1)
        
        if min_int < 0 or max_int > max_x_posible:
            st.error(f"⚠️ **Aviso:** Ambos valores del rango deben estar entre 0 y {max_x_posible}.")
        elif min_int > max_int:
            st.error("⚠️ **Error lógico:** El valor mínimo (a) no puede ser mayor al valor máximo (b).")
        else:
            prob_rango = rv.cdf(max_int) - (rv.cdf(min_int - 1) if min_int > 0 else 0)
            st.info(f"La probabilidad de sacar entre **{min_int}** y **{max_int}** éxitos es del **{(prob_rango*100):.4f}%**.")

        with st.expander("Ver Tabla Completa de Probabilidades y Exportar"):
            df_dist = pd.DataFrame({
                "Número de Éxitos (x)": eje_x,
                "Probabilidad Exacta": [f"{p:.8f}" for p in pmf_values],
                "Porcentaje (%)": [f"{(p * 100):.6f}%" for p in pmf_values]
            })
            st.dataframe(df_dist, hide_index=True, use_container_width=True)
            csv = df_dist.to_csv(index=False).encode('utf-8')
            st.download_button("Descargar tabla en CSV", data=csv, file_name=f'hipergeometrica.csv', mime='text/csv', type="primary")

    with tab2:
        st.markdown("### Función de Masa de Probabilidad (PMF)")
        df_plot = pd.DataFrame({'Éxitos': eje_x, 'Probabilidad': pmf_values, 'Selección': ['Calculado' if i == x else 'Otros' for i in eje_x]})
        fig = px.bar(df_plot, x='Éxitos', y='Probabilidad', color='Selección',
                     color_discrete_map={'Calculado': '#e74c3c', 'Otros': '#3498db'})
        
        fig.add_vline(x=esperanza, line_width=2, line_dash="dash", line_color="#27ae60", 
                      annotation_text=f"Media (μ) = {esperanza:.2f}", annotation_position="top right")
        
        fig.update_layout(showlegend=False, template="plotly_white", xaxis_title="Éxitos en la muestra (x)", yaxis_title="Probabilidad")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("💡 *La línea verde punteada muestra la Esperanza Matemática (el promedio esperado).*")
        
        if max_x_posible > 1000:
            st.warning("⚠️ La gráfica está truncada a los primeros 1000 valores para mantener la fluidez.")

    with tab3:
        st.markdown("### Teoría vs. Realidad (Simulación de Monte Carlo)")
        st.markdown("En este experimento virtual simularemos sacar la muestra miles de veces. Las barras oscuras son lo que dice la teoría matemática; las barras verdes son lo que pasó realmente en el simulador.")
        simulaciones = st.number_input("Número de simulaciones (Lanzamientos virtuales)", value=5000, step=100)
        
        if simulaciones < 100:
            st.error("⚠️ Agrega más simulaciones para apreciar la Ley de los Grandes Números.")
        elif simulaciones > 2000000:
            st.error("⚠️ Mantén las simulaciones por debajo de 2,000,000 para no saturar el servidor.")
        else:
            resultados_simulados = rv.rvs(size=simulaciones)
            conteo_empirico = pd.Series(resultados_simulados).value_counts(normalize=True).sort_index()
            ejes_sim = np.union1d(eje_x, conteo_empirico.index)
            
            df_montecarlo = pd.DataFrame({
                'Éxitos': ejes_sim,
                'Teórica': [rv.pmf(i) for i in ejes_sim],
                'Simulada': [conteo_empirico.get(i, 0) for i in ejes_sim]
            })

            fig2 = go.Figure()
            fig2.add_trace(go.Bar(x=df_montecarlo['Éxitos'], y=df_montecarlo['Teórica'], name='Probabilidad Teórica', marker_color='#2c3e50'))
            fig2.add_trace(go.Bar(x=df_montecarlo['Éxitos'], y=df_montecarlo['Simulada'], name='Resultados del Simulador', marker_color='#18bc9c', opacity=0.8))
            
            fig2.update_layout(barmode='group', template='plotly_white', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            st.plotly_chart(fig2, use_container_width=True)

    with tab4:
        st.markdown("### 1. Probabilidad Exacta $P(X=x)$")
        st.markdown("Calcula la probabilidad de obtener *exactamente* el número de éxitos que solicitaste en la muestra.")
        st.latex(r"P(X = x) = \frac{\binom{K}{x} \binom{N-K}{n-x}}{\binom{N}{n}}")
        st.latex(rf"P(X = {x}) = \frac{{\binom{{{K}}}{{{x}}} \binom{{{N}-{K}}}{{{n}-{x}}}}}{{\binom{{{N}}}{{{n}}}}} = {prob_exacta:.6f}")
        
        st.divider()
        
        st.markdown("### 2. Esperanza Matemática (Media, $\mu$)")
        st.markdown("💡 **Interpretación:** Es el valor promedio que esperarías obtener si repitieras este experimento (sacar la muestra) muchísimas veces.")
        st.latex(r"\mu = n \cdot \left( \frac{K}{N} \right)")
        st.latex(rf"\mu = {n} \cdot \left( \frac{{{K}}}{{{N}}} \right) = {esperanza:.4f}")
        st.metric("Esperanza matemática (μ)", f"{esperanza:.4f}")
        
        st.divider()
        
        st.markdown("### 3. Varianza ($\sigma^2$)")
        st.markdown("💡 **Interpretación:** Mide qué tan dispersos van a estar tus resultados reales respecto a la media. Una varianza alta significa mucha incertidumbre; una baja significa que tus resultados casi siempre estarán pegados a la media.")
        st.latex(r"\sigma^2 = n \cdot \left( \frac{K}{N} \right) \cdot \left( \frac{N-K}{N} \right) \cdot \left( \frac{N-n}{N-1} \right)")
        st.latex(rf"\sigma^2 = {n} \cdot \left( \frac{{{K}}}{{{N}}} \right) \cdot \left( \frac{{{N}-{K}}}{{{N}}} \right) \cdot \left( \frac{{{N}-{n}}}{{{N}-1}} \right) = {varianza:.4f}")
        st.metric("Varianza (σ²)", f"{varianza:.4f}")