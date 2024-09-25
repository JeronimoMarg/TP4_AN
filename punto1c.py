import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

distancias = np.array([0, 50, 100, 150, 200, 250, 300])
alturas = np.array([10, 60, 55, 70, 40, 50, 30])

# interpolacion con splines (cubicos)
spline_cubico = CubicSpline(distancias, alturas)

x_vals = np.linspace(0, 300, 400)
y_vals_spline = spline_cubico(x_vals)

plt.figure(figsize=(8, 6))
plt.plot(x_vals, y_vals_spline, label="Curva interpolada con Splines cúbicos", color="green")
plt.scatter(distancias, alturas, color="red", label="Puntos críticos")
plt.title("Interpolación de la montaña rusa con Splines cúbicos")
plt.xlabel("Distancia (m)")
plt.ylabel("Altura (m)")
plt.legend()
plt.grid(True)
plt.show()
