
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
# 1. Cargar el dataset Iris (o cualquier otro CSV)
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

# Añadir la columna objetivo para el análisis
df['species'] = iris.target_names[iris.target]

print("Primeras 5 filas del DataFrame:")
print(df.head())
print("\nInformación general del DataFrame:")
df.info()
# --- Paso 3: Análisis Descriptivo ---

print("\nEstadísticas Descriptivas (Resumen):")
print(df.describe())

print("\nConteo de Valores Únicos por Especie:")
print(df['species'].value_counts())

print("\nMedia de las Medidas por Especie:")
print(df.groupby('species').mean())

# Asegúrate de que estas librerías estén importadas al principio:
# import matplotlib.pyplot as plt
# import seaborn as sns

# --- Paso 4: Visualización de Datos ---

# 1. Histogramas y Curvas de Densidad (para ver la distribución de cada característica)
df.hist(figsize=(10, 8))
plt.suptitle('Distribución de Características del Dataset Iris', y=1.02)
plt.show()

# 2. Boxplots (para comparar la distribución entre especies)
plt.figure(figsize=(12, 6))
# Boxplot de la longitud del pétalo, diferenciado por la especie
sns.boxplot(x='species', y='petal length (cm)', data=df)
plt.title('Boxplot de Longitud del Pétalo por Especie')
plt.show()

# 3. Pairplot (Matriz de Gráficos de Dispersión)
# Esta es una herramienta PODEROSA para EDA
sns.pairplot(df, hue='species', diag_kind='kde')
plt.suptitle('Pairplot del Dataset Iris, coloreado por Especie', y=1.02)
plt.show()