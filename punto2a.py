import numpy as np
import random
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

# Datos de la montaña rusa
distancias = np.array([0, 50, 100, 150, 200, 250, 300])
alturas = np.array([10, 60, 55, 70, 40, 50, 30])
splines = CubicSpline(distancias, alturas)

# Parámetros del algoritmo genético
num_soportes = 20
min_dist = 10
max_dist = 20
longitud_pista = 300
tamano_poblacion = 50
num_generaciones = 200
tasa_mutacion = 0.1

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
         # si se pasa de la longitud y me faltan generar entonces intento de nuevo
        return generar_individuo()
    return individuo

# Función de aptitud
def evaluar_individuo(individuo):
    # evalua la altura en el punto determinado de separacion dentro de la pista
    return sum(splines(x) for x in individuo)

# Cruzamiento
def crossover(ind1, ind2):
    # el punto de cruce se determina aleatoriamente
    punto_cruce = random.randint(1, num_soportes - 1)
    # el primer hijo tiene la primera parte del padre1 y la segunda del padre 2
    hijo1 = ind1[:punto_cruce] + ind2[punto_cruce:]
    # el segundo hijo al reves
    hijo2 = ind2[:punto_cruce] + ind1[punto_cruce:]
    # los devuelve ordenados 
    return sorted(hijo1), sorted(hijo2)

# Mutación
def mutar_individuo(individuo):
    i = random.randint(1, num_soportes - 2)  # No se muta el primero o último soporte
    # la mutacion cambia la posicion de un soporte generado aleatoriamente
    nueva_posicion = individuo[i] + random.randint(-5, 5)
    nueva_posicion = max(individuo[i - 1] + min_dist, min(nueva_posicion, individuo[i + 1] - min_dist))  # Respetar límites
    individuo[i] = nueva_posicion
    return sorted(individuo)

# Selección por torneo
def seleccion_torneo(poblacion, k=3):
    # agarra el individuo que menos material necesita dentro de un grupo de 3 aleatoriamente elegidos
    seleccionados = random.sample(poblacion, k)
    mejor_individuo = min(seleccionados, key=lambda ind: evaluar_individuo(ind))
    return mejor_individuo

# Algoritmo genético
def algoritmo_genetico():
    # Generar población inicial
    # Genera 50 arrays de 20 individuos
    poblacion = [generar_individuo() for _ in range(tamano_poblacion)]
    
    for generacion in range(num_generaciones):
        nueva_poblacion = []
        
        # Crear nueva población con crossover y mutación
        while len(nueva_poblacion) < tamano_poblacion:
            padre1 = seleccion_torneo(poblacion)
            padre2 = seleccion_torneo(poblacion)
            hijo1, hijo2 = crossover(padre1, padre2)
            
            # Aplicar mutación
            if random.random() < tasa_mutacion:
                hijo1 = mutar_individuo(hijo1)
            if random.random() < tasa_mutacion:
                hijo2 = mutar_individuo(hijo2)
            
            nueva_poblacion.append(hijo1)
            nueva_poblacion.append(hijo2)
        
        # Reemplazar la población
        poblacion = nueva_poblacion
        
        # Evaluar el mejor individuo en esta generación
        mejor_individuo = min(poblacion, key=lambda ind: evaluar_individuo(ind))
        mejor_aptitud = evaluar_individuo(mejor_individuo)
        print(f"Generación {generacion+1}: Mejor aptitud = {mejor_aptitud}")
    
    # Devolver el mejor individuo encontrado
    mejor_individuo_final = min(poblacion, key=lambda ind: evaluar_individuo(ind))
    return mejor_individuo_final

# Ejecutar el algoritmo genético
mejor_solucion = algoritmo_genetico()
print("Mejor solución encontrada:", mejor_solucion)
print("Altura total de los soportes:", evaluar_individuo(mejor_solucion))

# Graficar la mejor solución
x_vals = np.linspace(0, longitud_pista, 400)
y_vals = splines(x_vals)

plt.figure(figsize=(8, 6))
plt.plot(x_vals, y_vals, label="Curva de la montaña rusa")
plt.scatter(mejor_solucion, splines(mejor_solucion), color="red", label="Soportes")
plt.title("Mejor colocación de soportes utilizando algoritmo genético")
plt.xlabel("Distancia (m)")
plt.ylabel("Altura (m)")
plt.legend()
plt.grid(True)
plt.show()
