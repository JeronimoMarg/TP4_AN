import numpy as np
import matplotlib.pyplot as plt

# Datos de la tabla
distancias = np.array([0, 50, 100, 150, 200, 250, 300])
alturas = np.array([10, 60, 55, 70, 40, 50, 30])
#Inciso A
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
#Inciso
# Calcular el número de condición de la matriz de Vandermonde
numero_condicion = np.linalg.cond(vandermonde_matrix)

print(f"Número de condición de la matriz de Vandermonde: {numero_condicion:.2e}")

# Función para calcular el polinomio de Lagrange
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

# Generar puntos para graficar
x_vals = np.linspace(0, 300, 400)
y_vals = [lagrange(x, distancias, alturas) for x in x_vals]

# Graficar los puntos críticos y la curva interpolada
plt.figure(figsize=(8, 6))
plt.plot(x_vals, y_vals, label="Curva interpolada con Lagrange", color="blue")
plt.scatter(distancias, alturas, color="red", label="Puntos críticos")
plt.title("Interpolación de Lagrange de la montaña rusa")
plt.xlabel("Distancia (m)")
plt.ylabel("Altura (m)")
plt.legend()
plt.grid(True)
plt.show()