import numpy as np
import random
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

# Datos de la montaña rusa
distancias = np.array([0, 50, 100, 150, 200, 250, 300])
alturas = np.array([10, 60, 55, 70, 40, 50, 30])

# Splines cúbicos
splines = CubicSpline(distancias, alturas)

# Derivadas del spline cúbico
spl_derivada_1 = splines.derivative(1)
spl_derivada_2 = splines.derivative(2)

# Polinomio de grado 6 usando matriz de Vandermonde
vandermonde_matrix = np.vander(distancias)
coef_polinomio = np.linalg.solve(vandermonde_matrix, alturas)

# Función polinómica de grado 6
def polinomio_vandermonde(x, coef):
    return np.polyval(coef, x)

# Parámetros del algoritmo genético
num_soportes = 20
min_dist = 10
max_dist = 20
longitud_pista = 300
tamano_poblacion = 50
num_generaciones = 50
tasa_mutacion = 0.1

# Función para calcular la curvatura
def calcular_curvatura(x):
    return np.abs(spl_derivada_2(x)) / (1 + spl_derivada_1(x)**2)**(3/2)

# Derivar el polinomio de Vandermonde (grado 6)
def derivada_polinomio(coef, n_derivada):
    deriv_coef = np.polyder(coef, m=n_derivada)
    return deriv_coef

# Curvatura para el polinomio de grado 6
def calcular_curvatura_vandermonde(x, coef):
    y_primera_derivada = np.polyval(derivada_polinomio(coef, 1), x)
    y_segunda_derivada = np.polyval(derivada_polinomio(coef, 2), x)
    curvatura = np.abs(y_segunda_derivada) / (1 + y_primera_derivada**2)**(3/2)
    return curvatura

# Función para generar un individuo
def generar_individuo():
    individuo = [0]  # Comenzamos en 0
    while len(individuo) < num_soportes:
        posible_posicion = individuo[-1] + random.randint(min_dist, max_dist)
        if posible_posicion <= longitud_pista:
            individuo.append(posible_posicion)
        else:
            break
    if len(individuo) < num_soportes:
        return generar_individuo()
    return individuo

# Modificar la función de evaluación para incluir curvatura con Vandermonde
def evaluar_individuo(individuo, metodo='spline'):
    if metodo == 'spline':
        alturas_soportes = np.array([splines(x) for x in individuo])
        curvaturas_soportes = np.array([calcular_curvatura(x) for x in individuo])
    else:
        alturas_soportes = np.array([polinomio_vandermonde(x, coef_polinomio) for x in individuo])
        curvaturas_soportes = np.array([calcular_curvatura_vandermonde(x, coef_polinomio) for x in individuo])
    
    promedio_ponderado = np.sum(alturas_soportes * curvaturas_soportes) / np.sum(curvaturas_soportes)
    return promedio_ponderado


# Función para ajustar las distancias entre soportes
def ajustar_distancias(individuo):
    for i in range(1, len(individuo)):
        # Si la distancia es menor que min_dist, ajustamos
        if individuo[i] - individuo[i - 1] < min_dist:
            individuo[i] = individuo[i - 1] + min_dist
        # Si la distancia es mayor que max_dist, ajustamos
        elif individuo[i] - individuo[i - 1] > max_dist:
            individuo[i] = individuo[i - 1] + max_dist
        # Si excedemos la longitud de la pista, regeneramos el individuo
        if individuo[i] > longitud_pista:
            return generar_individuo()
    return individuo

# Cruzamiento con corrección de restricciones
def crossover(ind1, ind2):
    punto_cruce = random.randint(1, num_soportes - 2)
    hijo1 = sorted(ind1[:punto_cruce] + ind2[punto_cruce:])
    hijo2 = sorted(ind2[:punto_cruce] + ind1[punto_cruce:])
    
    # Ajustar distancias de los hijos para respetar las restricciones
    hijo1 = ajustar_distancias(hijo1)
    hijo2 = ajustar_distancias(hijo2)
    
    return hijo1, hijo2

# Mutación con corrección de restricciones
def mutar_individuo(individuo):
    i = random.randint(1, num_soportes - 2)  # No se muta el primero o último soporte
    nueva_posicion = individuo[i] + random.randint(-5, 5)
    
    # Respetar las restricciones de separación
    nueva_posicion = max(individuo[i - 1] + min_dist, min(nueva_posicion, individuo[i + 1] - min_dist))
    
    # Si la nueva posición respeta los límites máximos
    if nueva_posicion - individuo[i - 1] > max_dist or individuo[i + 1] - nueva_posicion > max_dist:
        return mutar_individuo(individuo)  # Volver a mutar si no respeta las distancias
    else:
        individuo[i] = nueva_posicion
    
    return sorted(individuo)

# Selección por torneo
def seleccion_torneo(poblacion, k=3, metodo='spline'):
    seleccionados = random.sample(poblacion, k)
    mejor_individuo = min(seleccionados, key=lambda ind: evaluar_individuo(ind, metodo))
    return mejor_individuo

# Algoritmo genético
def algoritmo_genetico(metodo='spline'):
    # Generar población inicial
    poblacion = [generar_individuo() for _ in range(tamano_poblacion)]
    
    for generacion in range(num_generaciones):
        nueva_poblacion = []
        
        # Crear nueva población con crossover y mutación
        while len(nueva_poblacion) < tamano_poblacion:
            padre1 = seleccion_torneo(poblacion, metodo=metodo)
            padre2 = seleccion_torneo(poblacion, metodo=metodo)
            hijo1, hijo2 = crossover(padre1, padre2)
            
            # Aplicar mutación
            if random.random() < tasa_mutacion:
                hijo1 = mutar_individuo(hijo1)
            if random.random() < tasa_mutacion:
                hijo2 = mutar_individuo(hijo2)
            
            nueva_poblacion.append(hijo1)
            nueva_poblacion.append(hijo2)
        
        # Reemplazar la población
        poblacion = nueva_poblacion[:tamano_poblacion]  # Limitamos el tamaño de la población
        
        # Evaluar el mejor individuo en esta generación
        mejor_individuo = min(poblacion, key=lambda ind: evaluar_individuo(ind, metodo=metodo))
        mejor_aptitud = evaluar_individuo(mejor_individuo, metodo=metodo)
        print(f"Generación {generacion+1}: Mejor aptitud ({metodo}) = {mejor_aptitud}")
    
    # Devolver el mejor individuo encontrado
    mejor_individuo_final = min(poblacion, key=lambda ind: evaluar_individuo(ind, metodo=metodo))
    return mejor_individuo_final

# Ejecutar el algoritmo genético y graficar los mejores resultados

# 1. Ejecución del algoritmo genético para Splines Cúbicos
print("Ejecutando para Splines Cúbicos...")
mejor_solucion_spline = algoritmo_genetico(metodo='spline')
print("Mejor solución (Spline):", mejor_solucion_spline)
print("Promedio ponderado de las alturas con curvatura (Spline):", evaluar_individuo(mejor_solucion_spline, metodo='spline'))

# 2. Ejecución del algoritmo genético para Polinomio de Grado 6 (Vandermonde)
print("\nEjecutando para Polinomio de Grado 6 (Vandermonde)...")
mejor_solucion_vandermonde = algoritmo_genetico(metodo='vandermonde')
print("Mejor solución (Vandermonde):", mejor_solucion_vandermonde)
print("Promedio ponderado de las alturas con curvatura (Vandermonde):", evaluar_individuo(mejor_solucion_vandermonde, metodo='vandermonde'))

# 3. Graficar los resultados
x_vals = np.linspace(0, longitud_pista, 400)
y_spline = splines(x_vals)
y_vandermonde = polinomio_vandermonde(x_vals, coef_polinomio)

plt.figure(figsize=(12, 8))

# Curva de splines cúbicos
plt.plot(x_vals, y_spline, label='Splines Cúbicos', color='blue')

# Curva de polinomio de grado 6 (Vandermonde)
plt.plot(x_vals, y_vandermonde, label='Polinomio Grado 6 (Vandermonde)', color='red')


# Graficar las mejores soluciones para ambos métodos
soportes_spline_y = [splines(x) for x in mejor_solucion_spline]
soportes_vandermonde_y = [polinomio_vandermonde(x, coef_polinomio) for x in mejor_solucion_vandermonde]

# Soportes - Splines Cúbicos
plt.scatter(mejor_solucion_spline, soportes_spline_y, color='blue', marker='o', s=50, zorder=5, label='Soportes (Spline)')

# Soportes - Polinomio de Vandermonde
plt.scatter(mejor_solucion_vandermonde, soportes_vandermonde_y, color='red', marker='o', s=50, zorder=5, label='Soportes (Vandermonde)')

# Título y leyendas
plt.title('Comparación de las mejores soluciones: Splines cúbicos vs Polinomio de grado 6 (Vandermonde)')
plt.xlabel('Distancia (m)')
plt.ylabel('Altura (m)')
plt.legend()
plt.grid(True)
plt.show()
