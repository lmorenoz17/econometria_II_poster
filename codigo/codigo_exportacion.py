# %% Importación de paquetes ============================

#Rutas relativas
from pathlib import Path

#Módulos de numpy, pandas, matplotlib y spicy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import jarque_bera, probplot

#Modulos de statsmodels
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller, kpss

# %% Cargar bases de datos en python usando rutas relativas =========================

# Ruta del directorio raíz
BASE_DIR = Path(__file__).resolve().parents[2]

# Ruta del directorio datos
DATA_DIR = BASE_DIR / "datos"

# Ruta de la base de datos
ruta_imp = DATA_DIR / "base_de_datos.xlsx"

# %% ===================================
# Primera serie: Importaciones Colombia 1991-2025
# ======================================
imp_base = pd.read_excel(
    ruta_imp,
    skiprows=1,      # Salta la primera fila de encabezado de texto del Excel
    header=None,
    names=["imp_colombia"]
)
# Ver el tipo de objeto de la base de datos (Pandas.DataFrame)
print(type(imp_base))

# Ver primeras y últimas observaciones de la base de datos
print(imp_base.head()) # Primeras observaciones
print(imp_base.tail()) # Últimas observaciones 

# %% Creación del índice temporal de las series

# Creación del índice temporal de la serie de tiempo de "importaciones Colombia"
fechas_imp_base = pd.date_range(
    start="1991-01-31",
    periods=len(imp_base),
    freq="ME"        
)

# Agregar el índice temporal a la base de datos
imp_base.index = fechas_imp_base

# Ver el tipo de objeto
print(type(imp_base))

# Ver primeras y últimas observaciones
print(imp_base.head()) # Primeras observaciones
print(imp_base.tail()) # Últimas observaciones

# %% Creación de la serie de tiempo de "importaciones Colombia"

# La nueva serie de tiempo se va a llamar "imp_serie" y va a tener valores numéricos
imp_serie = imp_base["imp_colombia"].copy()
imp_serie = pd.to_numeric(imp_serie, errors="coerce")
imp_serie = imp_serie.dropna() # Borrar missing values

# Ver el principio y final de la serie de tiempo
print(imp_serie.head())
print(imp_serie.tail())

# # El tipo de objeto de la base de datos ahora es un Pandas.Series
print(type(imp_serie))

# Ver algunas estadísticas descriptivas de la serie de tiempo 
print(imp_serie.describe())

# %% =========================
# Paso 1: Identificación
# ============================

# Gráfica de la serie de tiempo "importaciones Colombia"
plt.figure(figsize=(10, 5))
plt.plot(imp_serie)
plt.title("Importaciones de Colombia (1991-2025)")
plt.xlabel("Fecha")
plt.ylabel("Valor")
plt.grid(True)
plt.show()

# %% FAC y FACP comparativas para 12, 24 y 36 rezagos (Serie Original)

lags_lista = [12, 24, 36]
fig, axes = plt.subplots(3, 2, figsize=(14, 10))

for i, num_lags in enumerate(lags_lista):
    # Gráfica FAC
    plot_acf(
        imp_serie,
        lags=num_lags,
        alpha=0.05,
        bartlett_confint=False,
        ax=axes[i, 0]
    )
    axes[i, 0].set_title(f"FAC de la Serie Original ({num_lags} rezagos / {num_lags//12} año{'s' if num_lags>12 else ''})")
    
    # Gráfica FACP
    plot_pacf(
        imp_serie,
        lags=num_lags,
        alpha=0.05,
        ax=axes[i, 1]
    )
    axes[i, 1].set_title(f"FACP de la Serie Original ({num_lags} rezagos / {num_lags//12} año{'s' if num_lags>12 else ''})")

plt.tight_layout()
plt.show()


#%% Tests de Raíz Unitaria 

# Test de Augmented Dickey Fuller (ADF)
adf_result = adfuller(imp_serie) # Test de Dickey Fuller sin intercepto y tendencia determínistica

# Nota: En el test de ADF si no rechazo la H0 la serie no es estacionaria
#       y si rechazo la H0 la serie es estacionaria. 

print("=== Test ADF ===")
print("Estadístico ADF:", adf_result[0])
print("p-valor:", adf_result[1])
print("Rezagos usados:", adf_result[2]) 
print("Observaciones:", adf_result[3])
print("Valores críticos:")
for nivel, valor in adf_result[4].items():
    print(f"{nivel}: {valor}")

if adf_result[1] < 0.05:
    print("ADF: Rechazamos H0. Según el test, la serie es estacionaria.")
else:
    print("ADF: No rechazamos H0. Según el test, la serie no es estacionaria.")

# Test KPSS
kpss_result = kpss(imp_serie, regression="c", nlags="auto")

# Nota: En prueba KPSS se interpreta al contario que una prueba ADF.
#       Si no rechazo la H0 la serie es estacionaria
#       y si rechazo la H0 la serie es no estacionaria. 

print("\n=== Test KPSS ===")
print("Estadístico KPSS:", kpss_result[0])
print("p-valor:", kpss_result[1])
print("Rezagos usados:", kpss_result[2])
print("Valores críticos:")
for nivel, valor in kpss_result[3].items():
    print(f"{nivel}: {valor}")

if kpss_result[1] < 0.05:
    print("KPSS: rechazamos H0. Según el test, la serie es no estacionaria.")
else:
    print("KPSS: no rechazamos H0. Según el test, La serie es estacionaria.")

# Nota: Según los resultados de la prueba ADF y KPSS, hay que 
#       diferenciar la serie. 
# %% Serie diferenciada
expo_serie_diff = expo_serie.diff().dropna()

# Gráfica de la serie de tiempo de la "diferencia exportaciones tradicionales"
plt.figure(figsize=(10, 5))
plt.plot(expo_serie_diff)
plt.title("Serie diferenciada")
plt.xlabel("Fecha")
plt.ylabel("Valor")
plt.grid(True)
plt.show()

# %% Transformación a Tasa de Crecimiento (Log-Diferencia)
imp_serie_log_diff = np.log(imp_serie).diff().dropna()

# Gráfica de la tasa de crecimiento estacionaria
plt.figure(figsize=(10, 5))
plt.plot(imp_serie_log_diff, color="navy")
plt.title("Tasa de Crecimiento Mensual de Importaciones (1991-2025)")
plt.xlabel("Fecha")
plt.ylabel("Tasa de variación")
plt.grid(True)
plt.show()

# %% FAC y FACP de la Tasa de Crecimiento
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

plot_acf(
    imp_serie_log_diff,
    lags=24,
    alpha=0.05,
    bartlett_confint=False,
    ax=axes[0]
)
axes[0].set_title("FAC de la Tasa de Crecimiento")

plot_pacf(
    imp_serie_log_diff,
    lags=24,
    alpha=0.05,
    ax=axes[1]
)
axes[1].set_title("FACP de la Tasa de Crecimiento")

plt.tight_layout()
plt.show()

# %% Identificación del modelo usando FAC y FACP

# La FAC muestra que solo la primera autocorrelación es significativa.
# La FACP decae rápidamente.
# Por tanto, se propone inicialmente un modelo MA(1) sobre la serie
# transformada, equivalente a un ARIMA(0, 1, 1) sobre log(imp_serie).

# P.d. También se pueden usar criterios de información para la 
#      Selección de los ordenes p y q del modelo ARIMA. 
#      Para acá se uso uso el criterio de la FAC y la FACP

# %% =========================
# Paso 2: Estimación
# ============================

# Lo primero es crear un objeto de la clase SARIMAX 
# específicación del modelo 
modelo_SARIMA = SARIMAX(
    np.log(imp_serie),
    order=(0, 1, 1),
    seasonal_order=(2, 0, 1, 12),
    trend="n",
    enforce_stationarity=False,
    enforce_invertibility=False
)

# Objeto tipo "SARIMAX" del paquete statmodels
type(modelo_SARIMA)

#  Acá es donde se realiza la estimación del modelo MA(1)
estimacion_SARIMA = modelo_SARIMA.fit(disp=False)

# Nota: El método de estimación es máxima verosimilitud 
#       sobre la representación del modelo en un espacio de estados

# Objeto tipo "SARIMAXResultsWrapper"
type(estimacion_SARIMA)

# Se imprimen los resultados principales de la estimación
# E.g. Coeficientes y significancia de los coeficientes
print(estimacion_SARIMA.summary())
# %% =========================
# Paso 3: Validación de supuestos
# ============================

# Residuales del modelo
residuales = estimacion_SARIMA.resid.dropna()

# Los residuales son un objeto tipo Pandas.Series
type(residuales)

# Gráfica de los residuales (deberían comportarse como ruido blanco)
plt.figure(figsize=(10, 5))
plt.plot(residuales.iloc[1:-1])
plt.title("Residuales del modelo SARIMA")
plt.xlabel("Fecha")
plt.ylabel("Valor")
plt.grid(True)
plt.show()

# Descripción de los residuales 
print(residuales.describe())

# %% FAC y FACP de los residuales

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

plot_acf(
    residuales,
    lags=24,
    alpha=0.05,
    bartlett_confint=False,
    ax=axes[0]
)

axes[0].set_title("FAC de los residuales")

plot_pacf(
    residuales,
    lags=24,
    alpha=0.05,
    ax=axes[1]
)

axes[1].set_title("FACP de los residuales")

plt.tight_layout()
plt.show()


# %% Prueba Ljung-Box

ljung_box = acorr_ljungbox(
    residuales,
    lags=[6, 12, 18, 24],
    return_df=True
)

print("Prueba Ljung-Box")
print(ljung_box)

# H0: no hay autocorrelación en los residuos.
# Si p-value > 0.05, no se rechaza H0.

# %% FAC y FACP de los residuales al cuadrado

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

plot_acf(
    residuales**2,
    lags=24,
    alpha=0.05,
    bartlett_confint=False,
    ax=axes[0]
)

axes[0].set_title("FAC de los residuales al cuadrado")

plot_pacf(
    residuales**2,
    lags=24,
    alpha=0.05,
    ax=axes[1]
)

axes[1].set_title("FACP de los residuales al cuadrado")

plt.tight_layout()
plt.show()

# %% Prueba ARCH de heterocedasticidad

arch_test = het_arch(residuales, nlags=12) 

print("Prueba ARCH")
print("LM statistic:", arch_test[0])
print("LM p-value:", arch_test[1])

# H0: no hay efectos ARCH.
# Si p-value > 0.05, no se rechaza H0.

# %% Q-Q plot de los residuos

plt.figure(figsize=(6, 6))
probplot(residuales.iloc[1:-1], dist="norm", plot=plt)
plt.title("Q-Q plot de los residuos")
plt.grid(True)
plt.show()

# %% Prueba de normalidad Jarque-Bera

jb_test = jarque_bera(residuales.iloc[1:-1])

print("Prueba Jarque-Bera")
print("Jarque-Bera statistic:", jb_test.statistic)
print("Jarque-Bera p-value:", jb_test.pvalue)

# H0: los residuos no siguen una distribución normal.
# Si p-value < 0.05, se rechaza H0.
# %%
# %% Pronóstico a 12 pasos adelante

# Pronóstico 12 pasos adelante
pronostico_log = estimacion_SARIMA.get_forecast(steps=12)

# Pronóstico puntual e intervalos de predicción en logaritmos
pronostico_puntual_log = pronostico_log.predicted_mean
varianza_pronostico_log = pronostico_log.var_pred_mean
intervalos_log = pronostico_log.conf_int()

# Corrección por sesgo de Jensen al retransformar desde logaritmos
# E[y] = exp(mu + 0.5 * sigma^2)
pronostico_nivel = np.exp(pronostico_puntual_log + 0.5 * varianza_pronostico_log)
intervalos_nivel = np.exp(intervalos_log)

# Tabla resumida de pronósticos
tabla_pronostico = pd.DataFrame({
    "pronostico": pronostico_nivel,
    "limite_inferior": intervalos_nivel.iloc[:, 0],
    "limite_superior": intervalos_nivel.iloc[:, 1]
})

print(tabla_pronostico)

# %% Gráfica del pronóstico

plt.figure(figsize=(10, 5))
# Graficar últimos años de datos históricos para apreciar mejor el gráfico
plt.plot(imp_serie.tail(60), label="Datos históricos", color="blue")
plt.plot(pronostico_nivel, label="Pronóstico", color="orange")
plt.fill_between(
    pronostico_nivel.index,
    intervalos_nivel.iloc[:, 0],
    intervalos_nivel.iloc[:, 1],
    color="orange",
    alpha=0.2,
    label="Intervalo de confianza"
)
plt.title("Pronóstico de Importaciones (SARIMA)")
plt.xlabel("Fecha")
plt.ylabel("Valor")
plt.legend(loc="upper left")
plt.grid(True)
plt.show()

# %%