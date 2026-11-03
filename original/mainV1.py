#!/usr/bin/env python

import os
import sys
import optparse
import numpy as np
# import funzzy logic library
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
import tensorflow as tf
import random

tf.data.experimental.enable_debug_mode()


print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


densidad_vehicular = ctrl.Antecedent(np.arange(0, 10, 1), "densidad")
cambio_estado = ctrl.Consequent(np.arange(0, 1, 0.1), "estado")

# Crear una función para la definición de antececentes y consecuentes
def defuzzify():
    # Definición de las variables lingüísticas


    # Definición de las funciones de pertenencia
    densidad_vehicular['low'] = fuzz.trapmf(densidad_vehicular.universe, [0, 0, 3, 5])
    densidad_vehicular['medium'] = fuzz.trapmf(densidad_vehicular.universe, [3, 5, 7, 10])
    densidad_vehicular['high'] = fuzz.trapmf(densidad_vehicular.universe, [7, 10, 10, 10])

    cambio_estado['actual'] = fuzz.trimf(cambio_estado.universe, [0, 0, 0.5])

    cambio_estado['siguiente'] = fuzz.trimf(cambio_estado.universe, [0.5, 1, 1])

    # Definición de las reglas
    rule1 = ctrl.Rule(densidad_vehicular['low'], cambio_estado['actual'])
    rule2 = ctrl.Rule(densidad_vehicular['medium'] | densidad_vehicular['high'], cambio_estado['siguiente'])

    # Creación del controlador difuso
    traffic_ctrl = ctrl.ControlSystem([rule1, rule2])


    # Simulación
    traffic = ctrl.ControlSystemSimulation(traffic_ctrl)

    return traffic



# Función para actualizar las funciones de pertenencia de la lógica difusa
def update_membership_functions(low, medium, high):
    densidad_vehicular['low'] = fuzz.trapmf(densidad_vehicular.universe, [0, 0] + low[1:])
    densidad_vehicular['medium'] = fuzz.trapmf(densidad_vehicular.universe, low[1:] + high[:-1])
    densidad_vehicular['high'] = fuzz.trapmf(densidad_vehicular.universe, high[:-1] + [10, 10])

    rule1 = ctrl.Rule(densidad_vehicular['low'], cambio_estado['actual'])
    rule2 = ctrl.Rule(densidad_vehicular['medium'] | densidad_vehicular['high'], cambio_estado['siguiente'])

    traffic_ctrl = ctrl.ControlSystem([rule1, rule2])
    traffic = ctrl.ControlSystemSimulation(traffic_ctrl)

    return traffic


def array_list(array_num):
    num_list = []  # list
    # Iterating over the array
    for i in array_num:
        num_list.append(i)

    return num_list


# we need to import some python modules from the $SUMO_HOME/tools directory
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

from sumolib import checkBinary  # Checks for the binary in environ vars
import traci


def get_options():
    opt_parser = optparse.OptionParser()
    opt_parser.add_option("--nogui", action="store_true",
                          default=False, help="run the commandline version of sumo")
    options, args = opt_parser.parse_args()
    return options

def get_options():
    opt_parser = optparse.OptionParser()
    opt_parser.add_option("--nogui", action="store_true",
                          default=True, help="run the commandline version of sumo")  # Cambie 'default' a True
    options, args = opt_parser.parse_args()
    return options

# contains TraCI control loop
def get_density(interseccion_id, calle):
    vehiculos = traci.lane.getLastStepVehicleIDs(calle)
    # longitud = traci.edge.getLength(interseccion_id)
    longitud = traci.lane.getLength(calle)
    return len(vehiculos) / longitud


def ajustar_conjuntos_difusos(densidad, controlador, model):
    controlador.input['densidad'] = densidad
    controlador.compute()
    tiempo_difuso = controlador.output['estado']
    tiempo_difuso_ajustado = model.predict(np.array([[densidad]]))[0][0]
    tiempo_ajustado = max(min(tiempo_difuso_ajustado, cambio_estado.universe[-1]), cambio_estado.universe[0])
    return tiempo_ajustado


def create_windowed_data(data, window_size):
    X = []
    y = []

    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size])

    return np.array(X), np.array(y)


def run():
    intersecciones = traci.trafficlight.getIDList()

    controlador = defuzzify()
    calles = []
    # un ciclo que recorra todas las intersecciones
    for interseccion_id in intersecciones:
        # Obtener la lista de conexiones controladas por el semáforo
        conexiones = traci.trafficlight.getControlledLanes(interseccion_id)

        calles_temp = []
        for c in conexiones:
           # print(c)
            calle_id = c

            if calle_id not in calles_temp:
                calles_temp.append(calle_id)

        calles.append(calles_temp)
    steps = 0
    # Modificar la sección del bucle principal
    densidades_acumuladas = []
    num_time_steps = 10
    num_features = 1

    model = tf.keras.Sequential([
        tf.keras.layers.Conv1D(32, 3, activation='relu', input_shape=(num_time_steps, num_features)),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Conv1D(64, 3, activation='relu'),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1)
    ])


    # Construcción del modelo LSTM
 #   model = tf.keras.Sequential([
 #       tf.keras.layers.Embedding(20000, 128),
 #       tf.keras.layers.LSTM(128, dropout=0.2, recurrent_dropout=0.2),
 #       tf.keras.layers.Dense(1, activation='sigmoid')
 #   ])
    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        steps = steps + 1

        # Acumular densidades en cada paso
        densidades = []
        for i in range(len(intersecciones)):
            if interseccion_id!=230:
                interseccion_id =intersecciones[i]
                ci = calles[i]
                for c in ci:
                    densidad = get_density(interseccion_id, c)
                    if densidad > 0:
                        densidades.append(densidad)

        if densidades:
            densidades_acumuladas.append(np.mean(densidades))

        if steps % 300 == 0:
        #if steps < 0:
            #print(len(intersecciones))
            if steps % 3000 == 0:
            #if steps < 0: #para que nunca entre
                print("Conteo de pasos: ", steps)
                # Preparar los datos para el entrenamiento de la red neuronal
                #X_train = np.array(densidades_acumuladas[:-1]).reshape(-1, 1)
                #y_train = np.array(densidades_acumuladas[1:]).reshape(-1, 1)
                X_train, y_train = create_windowed_data(densidades_acumuladas[:-1], num_time_steps)
                X_train = X_train.reshape(-1, num_time_steps, num_features)
                y_train = y_train.reshape(-1, 1)


                # Entrenar la red neuronal LSTM
                tf.config.run_functions_eagerly(True)
                model.compile(loss='mean_squared_error', optimizer='adam', run_eagerly=True)
                model.fit(X_train, y_train, epochs=100, batch_size=64)

                # Predecir el próximo valor de densidad
                #next_density = model.predict(X_train[-1].reshape(1, 1, 1))[0][0]
                next_density = model.predict(X_train[-1].reshape(1, num_time_steps, num_features))[0][0]


                # Actualizar las funciones de pertenencia de la lógica difusa
                low = [0, 0, next_density / 2]
                medium = [0, next_density / 2, next_density]
                high = [next_density / 2, next_density, next_density]
                controlador = update_membership_functions(low, medium, high)


            # Continuar con el resto de la lógica para el control de semáforos

            #print(len(intersecciones))
            for i in range(len(intersecciones)):
                interseccion_id = intersecciones[i]
                ci = calles[i]
                densidades = []
                calles_densidades = []

                for c in ci:
                    densidad = get_density(interseccion_id, c)

                    if densidad > 0:
                        densidades.append(densidad)
                        calles_densidades.append(c)
                if (len(densidades) > 0):
                    #maximo = np.max(densidades)
                    #minimo = np.min(densidades)

                    s = interseccion_id
                    # Obtener la definición del semáforo

                    definicion_semaforos = traci.trafficlight.getAllProgramLogics(s)

                    indice_fase = traci.trafficlight.getPhase(s)
                    num_fases = len(definicion_semaforos[0].getPhases())

                    estado_semaforo = traci.trafficlight.getRedYellowGreenState(s)

                    for i in range(len(calles_densidades)):
                        # print(int(densidades[i]*100))
                        controlador.input['densidad'] = densidades[i] * 100
                        controlador.compute()



                        if (controlador.output['estado'] >= 0.5):
                            # if (desicion_ajustada >= 0.5):
                            cdn = calles_densidades[i]

                            indice_calle = ci.index(cdn)

                            if estado_semaforo[indice_calle] == 'r':
                                indice_fase = indice_fase + 1
                                if indice_fase >= num_fases:
                                    indice_fase = 0
                                traci.trafficlight.setPhase(s, indice_fase)

                                print(f"Calle gestionada numero  : {calles_densidades[i]} con densidad {densidades[i]}")
                                # break


    # Desconexión de SUMO

    traci.close()


# main entry point
if __name__ == "__main__":
    options = get_options()

    # check binary
    if options.nogui:
        sumoBinary = checkBinary('sumo')
    else:
        sumoBinary = checkBinary('sumo-gui')

    for _ in range(100):
        seed = random.randint(1, 10000)  # Genera una semilla aleatoria entre 1 y 10000
        queue_output_file = f"colaControl{seed}.xml"  # Modifica el nombre del archivo de salida para incluir la semilla

        traci.start([sumoBinary, "-c", "run.sumocfg",
                     "--tripinfo-output", "tripinfo.xml", "--queue-output", queue_output_file, "--seed", str(seed)])
        run()

    # # traci starts sumo as a subprocess and then this script connects and runs
    # traci.start([sumoBinary, "-c", "run.sumocfg",
    #              "--tripinfo-output", "tripinfo.xml","--queue-output","colaControl8890.xml","--seed","8890"])
    # run()
