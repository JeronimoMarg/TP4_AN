import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

# Datos de la tabla
distancias = np.array([0, 50, 100, 150, 200, 250, 300])
alturas = np.array([10, 60, 55, 70, 40, 50, 30])

# vandermonde
vandermonde_matrix = np.vander(distancias, increasing=True)
coeficientes = np.linalg.solve(vandermonde_matrix, alturas)
polinomio = np.poly1d(coeficientes[::-1])

# con splines
splines = CubicSpline(distancias, alturas)

# alturas para calcular
x_estimaciones = [75, 225]

# estimacion con vandermonde
altura_polinomio_75 = polinomio(75)
altura_polinomio_225 = polinomio(225)

# estimacion con splines
altura_spline_75 = splines(75)
altura_spline_225 = splines(225)

print(f"Estimaciones para 75 metros:")
print(f"- Altura con polinomio de grado 6: {altura_polinomio_75:.2f}")
print(f"- Altura con splines cúbicos: {altura_spline_75:.2f}")
print(f"\nEstimaciones para 225 metros:")
print(f"- Altura con polinomio de grado 6: {altura_polinomio_225:.2f}")
print(f"- Altura con splines cúbicos: {altura_spline_225:.2f}")

# GRAFICADO

x_vals = np.linspace(0, 300, 400)
y_vals_polinomio = polinomio(x_vals)
y_vals_splines = splines(x_vals)

plt.figure(figsize=(10, 6))

plt.plot(x_vals, y_vals_polinomio, label="Polinomio de grado 6", color="blue")

plt.plot(x_vals, y_vals_splines, label="Splines cúbicos", color="green")

plt.scatter(distancias, alturas, color="red", label="Puntos críticos")

plt.scatter([75, 225], [altura_polinomio_75, altura_polinomio_225], color="blue", marker="o", edgecolor="black", zorder=5, label="Estimaciones Polinomio")
plt.scatter([75, 225], [altura_spline_75, altura_spline_225], color="green", marker="o", edgecolor="black", zorder=5, label="Estimaciones Splines")

plt.title("Comparación de interpolación: Polinomio de grado 6 vs Splines cúbicos")
plt.xlabel("Distancia (m)")
plt.ylabel("Altura (m)")
plt.legend()
plt.grid(True)
plt.show()
