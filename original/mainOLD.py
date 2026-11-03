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

densidad_vehicular = ctrl.Antecedent(np.arange(0, 10, 1), "densidad")
cambio_estado = ctrl.Consequent(np.arange(0, 1, 0.1), "estado")

model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, input_dim=1, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])


# crear una función para la definición de antececentes y consecuentes
def defuzzify():
    # Definición de las variables lingüísticas

    # Definición de las funciones de pertenencia
    densidad_vehicular['low'] = fuzz.trimf(densidad_vehicular.universe, [0, 0, 5])
    densidad_vehicular['medium'] = fuzz.trimf(densidad_vehicular.universe, [0, 5, 10])
    densidad_vehicular['high'] = fuzz.trimf(densidad_vehicular.universe, [5, 10, 10])

    cambio_estado['actual'] = fuzz.trimf(cambio_estado.universe, [0, 0, 0.5])
    # cambio_estado['siguiente'] = fuzz.trimf(cambio_estado.universe, [0, 0.5, 1])
    cambio_estado['siguiente'] = fuzz.trimf(cambio_estado.universe, [0.5, 1, 1])

    # Definición de las reglas
    rule1 = ctrl.Rule(densidad_vehicular['low'], cambio_estado['actual'])
    rule2 = ctrl.Rule(densidad_vehicular['medium'], cambio_estado['siguiente'])
    rule3 = ctrl.Rule(densidad_vehicular['high'], cambio_estado['siguiente'])

    # Creación del controlador difuso
    traffic_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])

    # Simulación
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


# contains TraCI control loop
def get_density(interseccion_id, calle):
    vehiculos = traci.lane.getLastStepVehicleIDs(calle)
    # longitud = traci.edge.getLength(interseccion_id)
    longitud = traci.lane.getLength(calle)
    return len(vehiculos) / longitud


def ajustar_conjuntos_difusos(densidad, controlador):
    controlador.input['densidad'] = densidad
    controlador.compute()
    tiempo_difuso = controlador.output['estado']
    tiempo_difuso_ajustado = model.predict(np.array([[densidad]]))[0][0]
    tiempo_ajustado = max(min(tiempo_difuso_ajustado, cambio_estado.universe[-1]), cambio_estado.universe[0])
    return tiempo_ajustado


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
            print(c)
            calle_id = c

            if calle_id not in calles_temp:
                calles_temp.append(calle_id)

        calles.append(calles_temp)
    steps = 0

    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        steps = steps + 1
        #
        if steps % 100 == 0:
            print(len(intersecciones))
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
                    maximo = np.max(densidades)
                    minimo = np.min(densidades)

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

                        # definición de red neuronal
                        # Entrenar una red neuronal para ajustar los conjuntos difusos
                        X_train = np.random.uniform(low=0, high=100, size=(10000, 1))
                        y_train = controlador.output['estado']

                        model = tf.keras.Sequential([
                            tf.keras.layers.Dense(64, input_dim=1, activation='relu'),
                            tf.keras.layers.Dense(64, activation='relu'),
                            tf.keras.layers.Dense(1)
                        ])
                        model.compile(loss='mse', optimizer='adam')

                        X_train = np.asarray(X_train)
                        y_train = np.asarray(y_train)


                #        model.fit(X_train, y_train, epochs=100, batch_size=64)
                        # fin de entrenamiento de red neuronal

                        # desicion_ajustada=ajustar_conjuntos_difusos(densidades[i], controlador)

                        if (controlador.output['estado'] >= 0.5):
                            # if (desicion_ajustada >= 0.5):
                            cdn = calles_densidades[i]

                            indice_calle = ci.index(cdn)

                            if (estado_semaforo[indice_calle] == 'r'):
                                indice_fase = indice_fase + 1
                                if (indice_fase >= num_fases):
                                    indice_fase = 0
                                traci.trafficlight.setPhase(s, indice_fase)

                                print(f"Calle gestionada numero  : {calles_densidades[i]} con densidad {densidades[i]}")
                                # break
                    #    if (densidades[i]==maximo):
                    #        cdn=calles_densidades[i]
                    # indice_calle=calles.index(cdn)
                    # if (estado_semaforo[indice_calle]=='r'):
                    #    indice_fase=indice_fase+1
                    #    if (indice_fase>=num_fases):
                    #        indice_fase=0
                    #    traci.trafficlight.setPhase(s, indice_fase)
                    #    print(f"Calle gestionada numero  : {calles_densidades[i]} con densidad {densidades[i]}")
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

    # traci starts sumo as a subprocess and then this script connects and runs
    traci.start([sumoBinary, "-c", "run.sumocfg",
                 "--tripinfo-output", "tripinfo.xml","--queue-output","colaControl8890.xml","--seed","8890"])
    run()
