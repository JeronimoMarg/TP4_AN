import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

# Datos de la tabla
distancias = np.array([0, 50, 100, 150, 200, 250, 300])
alturas = np.array([10, 60, 55, 70, 40, 50, 30])

# ---------------------------
# 1. Interpolación con polinomio de grado 6
# ---------------------------
# Matriz de Vandermonde para un polinomio de grado 6
vandermonde_matrix = np.vander(distancias, increasing=True)

# Resolución del sistema de ecuaciones lineales para encontrar los coeficientes del polinomio
coeficientes = np.linalg.solve(vandermonde_matrix, alturas)

# Definir el polinomio a partir de los coeficientes obtenidos
polinomio = np.poly1d(coeficientes[::-1])

# ---------------------------
# 2. Interpolación con Splines cúbicos
# ---------------------------
splines = CubicSpline(distancias, alturas)

# ---------------------------
# 3. Estimación de la altura en 75 y 225 metros
# ---------------------------
x_estimaciones = [75, 225]

# Estimación con polinomio de grado 6
altura_polinomio_75 = polinomio(75)
altura_polinomio_225 = polinomio(225)

# Estimación con Splines cúbicos
altura_spline_75 = splines(75)
altura_spline_225 = splines(225)

# Mostrar las estimaciones
print(f"Estimaciones para 75 metros:")
print(f"- Altura con polinomio de grado 6: {altura_polinomio_75:.2f}")
print(f"- Altura con splines cúbicos: {altura_spline_75:.2f}")
print(f"\nEstimaciones para 225 metros:")
print(f"- Altura con polinomio de grado 6: {altura_polinomio_225:.2f}")
print(f"- Altura con splines cúbicos: {altura_spline_225:.2f}")

# ---------------------------
# 4. Graficar las curvas de interpolación
# ---------------------------
# Generar puntos para graficar la curva interpolada
x_vals = np.linspace(0, 300, 400)
y_vals_polinomio = polinomio(x_vals)
y_vals_splines = splines(x_vals)

plt.figure(figsize=(10, 6))

# Graficar la interpolación con polinomio de grado 6
plt.plot(x_vals, y_vals_polinomio, label="Polinomio de grado 6", color="blue")

# Graficar la interpolación con splines cúbicos
plt.plot(x_vals, y_vals_splines, label="Splines cúbicos", color="green")

# Graficar los puntos críticos
plt.scatter(distancias, alturas, color="red", label="Puntos críticos")

# Graficar las estimaciones
plt.scatter([75, 225], [altura_polinomio_75, altura_polinomio_225], color="blue", marker="o", edgecolor="black", zorder=5, label="Estimaciones Polinomio")
plt.scatter([75, 225], [altura_spline_75, altura_spline_225], color="green", marker="o", edgecolor="black", zorder=5, label="Estimaciones Splines")

# Configuraciones del gráfico
plt.title("Comparación de interpolación: Polinomio de grado 6 vs Splines cúbicos")
plt.xlabel("Distancia (m)")
plt.ylabel("Altura (m)")
plt.legend()
plt.grid(True)
plt.show()
