import numpy as np
import matplotlib.pyplot as plt

distancias = np.array([0, 50, 100, 150, 200, 250, 300])
alturas = np.array([10, 60, 55, 70, 40, 50, 30])

# matriz de vandermonde
vandermonde_matrix = np.vander(distancias, increasing=True)

# resolver el sistema
coeficientes = np.linalg.solve(vandermonde_matrix, alturas)

# esto genera el polinomio a traves de los coeficientes
polinomio = np.poly1d(coeficientes[::-1])

x_vals = np.linspace(0, 300, 400)
y_vals = polinomio(x_vals)

plt.figure(figsize=(8, 6))
plt.plot(x_vals, y_vals, label="Curva interpolada", color="blue")
plt.scatter(distancias, alturas, color="red", label="Puntos críticos")
plt.title("Interpolación de la montaña rusa con un polinomio de grado 6")
plt.xlabel("Distancia (m)")
plt.ylabel("Altura (m)")
plt.legend()
plt.grid(True)
plt.show()


