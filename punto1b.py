import numpy as np
import matplotlib.pyplot as plt 

distancias = np.array([0, 50, 100, 150, 200, 250, 300])
alturas = np.array([10, 60, 55, 70, 40, 50, 30])

vandermonde_matrix = np.vander(distancias, increasing=True)

# condicion de la matriz de vandermonde
numero_condicion = np.linalg.cond(vandermonde_matrix)

print(f"Número de condición de la matriz de Vandermonde: {numero_condicion}")

# hacer lagrange
def lagrange(x, x_points, y_points):
    n = len(x_points)
    result = 0
    for i in range(n):
        term = y_points[i]
        for j in range(n):
            if i != j:
                term *= (x - x_points[j]) / (x_points[i] - x_points[j])
        result += term
    return result

x_vals = np.linspace(0, 300, 400)
y_vals = [lagrange(x, distancias, alturas) for x in x_vals]

plt.figure(figsize=(8, 6))
plt.plot(x_vals, y_vals, label="Curva interpolada con Lagrange", color="blue")
plt.scatter(distancias, alturas, color="red", label="Puntos críticos")
plt.title("Interpolación de Lagrange de la montaña rusa")
plt.xlabel("Distancia (m)")
plt.ylabel("Altura (m)")
plt.legend()
plt.grid(True)
plt.show()
