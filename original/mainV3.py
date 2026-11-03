#!/usr/bin/env python
"""
Programa para gestionar semáforos usando un sistema difuso y una red neuronal para predecir el largo de cola.
La lógica original se conserva, pero se han incorporado mejoras para optimizar la corrida y la implementación de la red.
"""

import os
import sys
import optparse
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
import tensorflow as tf
import random

# Habilitar modo debug (puedes comentarlo si es necesario)
tf.data.experimental.enable_debug_mode()

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# --- DEFINICIÓN DE VARIABLES DIFUSAS ---
densidad_vehicular = ctrl.Antecedent(np.arange(0, 10, 1), "densidad")
cambio_estado = ctrl.Consequent(np.arange(0, 1, 0.1), "estado")


def defuzzify():
    """
    Define las funciones de pertenencia del sistema difuso
    y establece dos reglas:
      - Si la densidad es baja, se mantiene el estado actual.
      - Si la densidad es media o alta, se cambia al siguiente estado.
    """
    # Funciones de pertenencia para la densidad vehicular
    densidad_vehicular['low'] = fuzz.trapmf(densidad_vehicular.universe, [0, 0, 3, 5])
    densidad_vehicular['medium'] = fuzz.trapmf(densidad_vehicular.universe, [3, 5, 7, 10])
    densidad_vehicular['high'] = fuzz.trapmf(densidad_vehicular.universe, [7, 10, 10, 10])

    # Funciones de pertenencia para el cambio de estado (salida)
    cambio_estado['actual'] = fuzz.trimf(cambio_estado.universe, [0, 0, 0.5])
    cambio_estado['siguiente'] = fuzz.trimf(cambio_estado.universe, [0.5, 1, 1])

    # Definición de reglas
    rule1 = ctrl.Rule(densidad_vehicular['low'], cambio_estado['actual'])
    rule2 = ctrl.Rule(densidad_vehicular['medium'] | densidad_vehicular['high'], cambio_estado['siguiente'])

    # Creación del sistema difuso y su simulación
    traffic_ctrl = ctrl.ControlSystem([rule1, rule2])
    traffic = ctrl.ControlSystemSimulation(traffic_ctrl)
    return traffic


def update_membership_functions(low, medium, high):
    """
    Actualiza las funciones de pertenencia de la variable de entrada según las predicciones.
    Los argumentos "low", "medium" y "high" son listas que definen los puntos característicos.
    """
    # Actualiza cada función: se mantienen los extremos y se usan los nuevos valores para los puntos medios.
    densidad_vehicular['low'] = fuzz.trapmf(densidad_vehicular.universe, [0, 0] + low[1:])
    densidad_vehicular['medium'] = fuzz.trapmf(densidad_vehicular.universe, low[1:] + high[:-1])
    densidad_vehicular['high'] = fuzz.trapmf(densidad_vehicular.universe, high[:-1] + [10, 10])

    # Se redefinen las reglas de control
    rule1 = ctrl.Rule(densidad_vehicular['low'], cambio_estado['actual'])
    rule2 = ctrl.Rule(densidad_vehicular['medium'] | densidad_vehicular['high'], cambio_estado['siguiente'])

    traffic_ctrl = ctrl.ControlSystem([rule1, rule2])
    traffic = ctrl.ControlSystemSimulation(traffic_ctrl)
    return traffic


def create_windowed_data(data, window_size):
    """
    Crea ventanas (secuencias) de datos para entrenar la red neuronal.
    """
    X = []
    y = []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size])
    return np.array(X), np.array(y)


# --- CONFIGURACIÓN DE SUMO ---
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

from sumolib import checkBinary  # Verifica el binario de SUMO
import traci


def get_options():
    opt_parser = optparse.OptionParser()
    opt_parser.add_option("--nogui", action="store_true", default=True,
                          help="Run the commandline version of sumo")
    options, args = opt_parser.parse_args()
    return options


def get_density(interseccion_id, calle):
    """
    Calcula la densidad de vehículos en la calle (usando la longitud de la lane).
    """
    vehiculos = traci.lane.getLastStepVehicleIDs(calle)
    longitud = traci.lane.getLength(calle)
    return len(vehiculos) / longitud


# --- BUCLE DE SIMULACIÓN ---
def run():
    # Obtener lista de intersecciones
    intersecciones = traci.trafficlight.getIDList()
    controlador = defuzzify()  # Inicialmente crea el sistema difuso
    calles = []  # Lista de calles de cada intersección

    # Para cada intersección, obtener las lanes (sin duplicados)
    for interseccion_id in intersecciones:
        conexiones = traci.trafficlight.getControlledLanes(interseccion_id)
        calles_temp = []
        for c in conexiones:
            if c not in calles_temp:
                calles_temp.append(c)
        calles.append(calles_temp)

    steps = 0
    densidades_acumuladas = []
    num_time_steps = 10
    num_features = 1

    # Construir el modelo de red neuronal (usa una arquitectura CNN)
    model = tf.keras.Sequential([
        tf.keras.layers.Conv1D(32, 3, activation='relu', input_shape=(num_time_steps, num_features)),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Conv1D(64, 3, activation='relu'),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    # Compilación única del modelo
    model.compile(loss='mean_squared_error', optimizer='adam')

    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        steps += 1

        # Acumular densidades a lo largo de la simulación
        densidades = []
        for i in range(len(intersecciones)):
            # (Opcional: omitir intersección '230' si se desea)
            if intersecciones[i] != '230':
                inter_id = intersecciones[i]
                ci = calles[i]
                for c in ci:
                    d = get_density(inter_id, c)
                    if d > 0:
                        densidades.append(d)
        if densidades:
            densidades_acumuladas.append(np.mean(densidades))

        # Cada 300 pasos (y cada 3000 pasos se entrena el modelo)
        if steps % 300 == 0:
            if steps % 3000 == 0:
                print("Conteo de pasos:", steps)
                X_train, y_train = create_windowed_data(densidades_acumuladas[:-1], num_time_steps)
                X_train = X_train.reshape(-1, num_time_steps, num_features)
                y_train = y_train.reshape(-1, 1)
                early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
                model.fit(X_train, y_train, epochs=50, batch_size=64, callbacks=[early_stop], verbose=1)
                next_density = model.predict(X_train[-1].reshape(1, num_time_steps, num_features))[0][0]
                # Se definen nuevos parámetros para actualizar los conjuntos difusos.
                low = [0, 0, next_density / 2]
                medium = [0, next_density / 2, next_density]
                high = [next_density / 2, next_density, next_density]
                controlador = update_membership_functions(low, medium, high)

            # Control de semáforos para cada intersección
            for i in range(len(intersecciones)):
                interseccion_id = intersecciones[i]
                ci = calles[i]
                densidades_interseccion = []
                calles_densidades = []
                # Obtener densidad en cada calle de la intersección
                for c in ci:
                    d = get_density(interseccion_id, c)
                    if d > 0:
                        densidades_interseccion.append(d)
                        calles_densidades.append(c)
                if len(densidades_interseccion) > 0:
                    s = interseccion_id
                    # Obtener la definición del semáforo actual
                    definicion_semaforos = traci.trafficlight.getAllProgramLogics(s)
                    indice_fase = traci.trafficlight.getPhase(s)
                    num_fases = len(definicion_semaforos[0].getPhases())
                    estado_semaforo = traci.trafficlight.getRedYellowGreenState(s)
                    # Para cada calle de la intersección, aplicar el control difuso:
                    for j in range(len(calles_densidades)):
                        # Escalar la densidad (se multiplica por 100, según la lógica original)
                        controlador.input['densidad'] = densidades_interseccion[j] * 100
                        controlador.compute()
                        # Si el output indica que el estado debe cambiar (umbral 0.5), cambiar la fase
                        if controlador.output['estado'] >= 0.5:
                            indice_fase += 1
                            if indice_fase >= num_fases:
                                indice_fase = 0
                            traci.trafficlight.setPhase(s, indice_fase)
                            print(
                                f"Calle gestionada número: {calles_densidades[j]} con densidad {densidades_interseccion[j]}")

    traci.close()


# --- PUNTO DE ENTRADA PRINCIPAL ---
if __name__ == "__main__":
    options = get_options()
    # Seleccionar el binario de SUMO según si se usa GUI o no
    if options.nogui:
        sumoBinary = checkBinary('sumo')
    else:
        sumoBinary = checkBinary('sumo-gui')
    # Ejecutar 100 corridas de la simulación (se pueden ajustar)
    for _ in range(100):
        seed = random.randint(1, 10000)
        queue_output_file = f"colaControl{seed}.xml"
        traci.start([sumoBinary, "-c", "run.sumocfg",
                     "--tripinfo-output", "tripinfo.xml",
                     "--queue-output", queue_output_file,
                     "--seed", str(seed)])
        run()
