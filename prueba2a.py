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
num_generaciones = 100
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
        return generar_individuo()  # Si no logramos tener suficientes soportes, volvemos a generar
    return individuo

# Función de aptitud
def evaluar_individuo(individuo):
    return sum(splines(x) for x in individuo)

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

# cruza (crossover)
def crossover(ind1, ind2):
    punto_cruce = random.randint(1, num_soportes - 2)
    hijo1 = sorted(ind1[:punto_cruce] + ind2[punto_cruce:])
    hijo2 = sorted(ind2[:punto_cruce] + ind1[punto_cruce:])
    
    # Aajustar distancias si se violan las restricciones
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
def seleccion_torneo(poblacion, k=3):
    seleccionados = random.sample(poblacion, k)
    mejor_individuo = min(seleccionados, key=lambda ind: evaluar_individuo(ind))
    return mejor_individuo

#algoritmo genetico
def algoritmo_genetico():
    # poblacion inicial
    poblacion = [generar_individuo() for _ in range(tamano_poblacion)]
    
    #corre para todas las veces que definimos
    for generacion in range(num_generaciones):
        nueva_poblacion = []
        
        # se genera una nueva poblacion
        while len(nueva_poblacion) < tamano_poblacion:

            #torneo
            padre1 = seleccion_torneo(poblacion)
            padre2 = seleccion_torneo(poblacion)
            #crossover
            hijo1, hijo2 = crossover(padre1, padre2)
            #mutar
            if random.random() < tasa_mutacion:
                hijo1 = mutar_individuo(hijo1)
            if random.random() < tasa_mutacion:
                hijo2 = mutar_individuo(hijo2)
            
            #generacion de la poblacion nueva
            nueva_poblacion.append(hijo1)
            nueva_poblacion.append(hijo2)
        
        #reemplazar la poblacion
        poblacion = nueva_poblacion[:tamano_poblacion]
        
        #se fija el mejor individuo dentro de la poblacion y lo muestra en pantalla
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
