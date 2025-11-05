import numpy as np
import matplotlib.pyplot as plt

# --- 1. Generación de Datos Sintéticos ---
# Crear un conjunto de datos simple para la regresión

# Valores de la variable independiente (X)
X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# Valores de la variable dependiente (Y)
# Y = 2*X + 1 + ruido
Y = 2 * X + 1 + np.random.randn(10) * 1.5

# Visualización rápida para ver la relación lineal
plt.figure(figsize=(8, 6))
plt.scatter(X, Y, color='blue', label='Datos Originales')
plt.title('Datos Sintéticos para Regresión Lineal')
plt.xlabel('X (Variable Independiente)')
plt.ylabel('Y (Variable Dependiente)')
plt.legend()
plt.grid(True)
plt.show()

# Transformamos X a una matriz columna para cálculos matriciales
X = X.reshape(-1, 1) 
Y = Y.reshape(-1, 1)

print("X (forma):", X.shape)
print("Y (forma):", Y.shape)
#Creamos una columna de unos de la misma dimensión que X
columna_de_unos = np.ones_like(X)

# Concatenamos horizontalmente (hstack) la columna de unos y la matriz X original
# ¡Esta es la Matriz de Diseño que necesitamos!
X_diseño = np.hstack((columna_de_unos, X))

print("Primeras 5 filas de la Matriz de Diseño X_diseño (con la columna de unos):")
print(X_diseño[:5])
print("X_diseño (forma):", X_diseño.shape)
# Importamos np.linalg (álgebra lineal)
import numpy.linalg as la 

# --- 3. Solución vía Descomposición QR ---

# a) Realizar la Descomposición QR de la Matriz de Diseño
Q, R = la.qr(X_diseño)

print("\n--- Descomposición QR ---")
print(f"Matriz Q (Ortogonal, forma {Q.shape}):")
print(Q[:3])
print(f"Matriz R (Triangular Superior, forma {R.shape}):")
print(R)

# b) Calcular el lado derecho: Q^T * Y
# np.dot realiza el producto matricial.
QT_Y = np.dot(Q.T, Y)

# c) Resolver el sistema triangular R * beta = Q^T * Y
# Los resultados (beta) son los parámetros [b_0, b_1]
parametros_beta = la.solve(R, QT_Y)

# Asignar los parámetros
b_0 = parametros_beta[0, 0] # Intercepto
b_1 = parametros_beta[1, 0] # Pendiente

print("\n--- Parámetros del Modelo (Obtenidos por QR) ---")
print(f"Intercepción (b_0): {b_0:.4f}")
print(f"Pendiente (b_1): {b_1:.4f}")
# Asegúrate de tener 'import matplotlib.pyplot as plt' al inicio del script.

# --- 4. Visualización del Ajuste del Modelo ---

# a) Generar los valores predichos (Y_pred)
# Usamos la ecuación del modelo lineal: Y_pred = b_0 + b_1 * X
# Recuerda que X es el array original de la variable independiente.
Y_pred = b_0 + b_1 * X

# b) Configurar y generar el gráfico
plt.figure(figsize=(8, 6))

# Gráfico de dispersión de los datos originales
plt.scatter(X, Y, color='blue', label='Datos Originales (X vs Y)')

# Trazar la línea de regresión
# La línea va desde (X, Y_pred)
plt.plot(X, Y_pred, color='red', linewidth=3, 
         label=f'Línea de Regresión: Y = {b_1:.2f}X + {b_0:.2f}')

plt.title('Regresión Lineal Simple Ajustada por Descomposición QR')
plt.xlabel('X (Variable Independiente)')
plt.ylabel('Y (Variable Dependiente)')
plt.legend()
plt.grid(True)
plt.show()