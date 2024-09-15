import numpy as np
import matplotlib.pyplot as plt

# Datos de la tabla
distancias = np.array([0, 50, 100, 150, 200, 250, 300])
alturas = np.array([10, 60, 55, 70, 40, 50, 30])

# Matriz de Vandermonde para un polinomio de grado 6
vandermonde_matrix = np.vander(distancias, increasing=True)

# Resolución del sistema de ecuaciones lineales para encontrar los coeficientes del polinomio
coeficientes = np.linalg.solve(vandermonde_matrix, alturas)

# Definir el polinomio a partir de los coeficientes obtenidos
polinomio = np.poly1d(coeficientes[::-1])

# Generar puntos para graficar la curva interpolada
x_vals = np.linspace(0, 300, 400)
y_vals = polinomio(x_vals)

# Graficar los puntos críticos y la curva interpolada
plt.figure(figsize=(8, 6))
plt.plot(x_vals, y_vals, label="Curva interpolada", color="blue")
plt.scatter(distancias, alturas, color="red", label="Puntos críticos")
plt.title("Interpolación de la montaña rusa con un polinomio de grado 6")
plt.xlabel("Distancia (m)")
plt.ylabel("Altura (m)")
plt.legend()
plt.grid(True)
plt.show()


