#!/usr/bin/env python
from tqdm import tqdm
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
import tkinter as tk
from tkinter import ttk
import threading

tf.data.experimental.enable_debug_mode()

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
densidad_vehicular = ctrl.Antecedent(np.arange(0, 10, 1), "densidad")
cambio_estado = ctrl.Consequent(np.arange(0, 1, 0.1), "estado")

# Crear las variables de densidad para semáforos en rojo y verde
densidad_rojo = ctrl.Antecedent(np.linspace(0, 1, 100), 'rojo')
densidad_verde = ctrl.Antecedent(np.linspace(0, 1, 100), 'verde')

# Crear la variable de salida de densidad resultante
densidad_resultante = ctrl.Consequent(np.linspace(0, 1, 100), 'resultante')


# Crear una función para la definición de antececentes y consecuentes
def defuzzify():
    # Definir las funciones de membresía para densidad_rojo y densidad_verde
    niveles = ['muy_muy_bajo', 'muy_bajo', 'bajo', 'medio', 'alto', 'muy_alto', 'muy_muy_alto']
    for var in [densidad_rojo, densidad_verde]:
        var[niveles[0]] = fuzz.trimf(var.universe, [0, 0, 0.167])
        var[niveles[1]] = fuzz.trimf(var.universe, [0, 0.167, 0.334])
        var[niveles[2]] = fuzz.trimf(var.universe, [0.167, 0.334, 0.501])
        var[niveles[3]] = fuzz.trimf(var.universe, [0.334, 0.501, 0.668])
        var[niveles[4]] = fuzz.trimf(var.universe, [0.501, 0.668, 0.835])
        var[niveles[5]] = fuzz.trimf(var.universe, [0.668, 0.835, 1])
        var[niveles[6]] = fuzz.trimf(var.universe, [0.835, 1, 1])

    # Definir las funciones de membresía para densidad_resultante
    densidad_resultante['muy_muy_bajo'] = fuzz.trimf(densidad_resultante.universe, [0, 0, 0.167])
    densidad_resultante['muy_bajo'] = fuzz.trimf(densidad_resultante.universe, [0, 0.167, 0.334])
    densidad_resultante['bajo'] = fuzz.trimf(densidad_resultante.universe, [0.167, 0.334, 0.501])
    densidad_resultante['medio'] = fuzz.trimf(densidad_resultante.universe, [0.334, 0.501, 0.668])
    densidad_resultante['alto'] = fuzz.trimf(densidad_resultante.universe, [0.501, 0.668, 0.835])
    densidad_resultante['muy_alto'] = fuzz.trimf(densidad_resultante.universe, [0.668, 0.835, 1])
    densidad_resultante['muy_muy_alto'] = fuzz.trimf(densidad_resultante.universe, [0.835, 1, 1])

    # Definir las reglas de la lógica difusa
    rules = []

    # Para densidad media
    rules.append(ctrl.Rule(densidad_rojo[niveles[0]] & densidad_verde[niveles[0]], densidad_resultante['medio']))
    rules.append(ctrl.Rule(densidad_rojo[niveles[1]] & densidad_verde[niveles[1]], densidad_resultante['medio']))
    rules.append(ctrl.Rule(densidad_rojo[niveles[2]] & densidad_verde[niveles[2]], densidad_resultante['medio']))
    rules.append(ctrl.Rule(densidad_rojo[niveles[3]] & densidad_verde[niveles[3]], densidad_resultante['medio']))
    rules.append(ctrl.Rule(densidad_rojo[niveles[4]] & densidad_verde[niveles[4]], densidad_resultante['medio']))
    rules.append(ctrl.Rule(densidad_rojo[niveles[5]] & densidad_verde[niveles[5]], densidad_resultante['medio']))

    # Para densidad muy muy alta
    rules.append(ctrl.Rule(densidad_rojo[niveles[0]] & densidad_verde[niveles[6]], densidad_resultante['muy_muy_alto']))
    rules.append(ctrl.Rule(densidad_rojo[niveles[0]] & densidad_verde[niveles[5]], densidad_resultante['muy_muy_alto']))
    rules.append(ctrl.Rule(densidad_rojo[niveles[1]] & densidad_verde[niveles[6]], densidad_resultante['muy_muy_alto']))

    # Para densidad muy muy bajo
    rules.append(ctrl.Rule(densidad_rojo[niveles[6]] & densidad_verde[niveles[0]], densidad_resultante['muy_muy_bajo']))
    rules.append(ctrl.Rule(densidad_rojo[niveles[5]] & densidad_verde[niveles[0]], densidad_resultante['muy_muy_bajo']))
    rules.append(ctrl.Rule(densidad_rojo[niveles[6]] & densidad_verde[niveles[1]], densidad_resultante['muy_muy_bajo']))

    # Para densidad muy alta
    rules.append(ctrl.Rule(densidad_rojo[niveles[2]] & densidad_verde[niveles[5]], densidad_resultante['muy_alto']))
    rules.append(ctrl.Rule(densidad_rojo[niveles[1]] & densidad_verde[niveles[4]], densidad_resultante['muy_alto']))
    rules.append(ctrl.Rule(densidad_rojo[niveles[1]] & densidad_verde[niveles[5]], densidad_resultante['muy_alto']))

    rules.append(ctrl.Rule(densidad_rojo[niveles[0]] & densidad_verde[niveles[3]], densidad_resultante['muy_alto']))
    rules.append(ctrl.Rule(densidad_rojo[niveles[0]] & densidad_verde[niveles[4]], densidad_resultante['muy_alto']))

    rules.append(ctrl.Rule(densidad_rojo[niveles[2]] & densidad_verde[niveles[6]], densidad_resultante['muy_alto']))
    rules.append(ctrl.Rule(densidad_rojo[niveles[3]] & densidad_verde[niveles[6]], densidad_resultante['muy_alto']))

    # Para densidad alta
    rules.append(ctrl.Rule(densidad_rojo[niveles[1]] & densidad_verde[niveles[2]], densidad_resultante['alto']))
    rules.append(ctrl.Rule(densidad_rojo[niveles[1]] & densidad_verde[niveles[3]], densidad_resultante['alto']))
    rules.append(ctrl.Rule(densidad_rojo[niveles[2]] & densidad_verde[niveles[3]], densidad_resultante['alto']))
    rules.append(ctrl.Rule(densidad_rojo[niveles[2]] & densidad_verde[niveles[4]], densidad_resultante['alto']))
    rules.append(ctrl.Rule(densidad_rojo[niveles[3]] & densidad_verde[niveles[4]], densidad_resultante['alto']))
    rules.append(ctrl.Rule(densidad_rojo[niveles[3]] & densidad_verde[niveles[5]], densidad_resultante['alto']))
    rules.append(ctrl.Rule(densidad_rojo[niveles[4]] & densidad_verde[niveles[5]], densidad_resultante['alto']))

    rules.append(ctrl.Rule(densidad_rojo[niveles[0]] & densidad_verde[niveles[1]], densidad_resultante['alto']))
    rules.append(ctrl.Rule(densidad_rojo[niveles[0]] & densidad_verde[niveles[2]], densidad_resultante['alto']))

    rules.append(ctrl.Rule(densidad_rojo[niveles[4]] & densidad_verde[niveles[6]], densidad_resultante['alto']))
    rules.append(ctrl.Rule(densidad_rojo[niveles[5]] & densidad_verde[niveles[6]], densidad_resultante['alto']))

    # Para densidad muy baja
    rules.append(ctrl.Rule(densidad_rojo[niveles[5]] & densidad_verde[niveles[1]], densidad_resultante['muy_bajo']))
    rules.append(ctrl.Rule(densidad_rojo[niveles[4]] & densidad_verde[niveles[1]], densidad_resultante['muy_bajo']))
    rules.append(ctrl.Rule(densidad_rojo[niveles[5]] & densidad_verde[niveles[2]], densidad_resultante['muy_bajo']))

    rules.append(ctrl.Rule(densidad_rojo[niveles[4]] & densidad_verde[niveles[0]], densidad_resultante['muy_bajo']))
    rules.append(ctrl.Rule(densidad_rojo[niveles[3]] & densidad_verde[niveles[0]], densidad_resultante['muy_bajo']))

    rules.append(ctrl.Rule(densidad_rojo[niveles[6]] & densidad_verde[niveles[3]], densidad_resultante['muy_bajo']))
    rules.append(ctrl.Rule(densidad_rojo[niveles[6]] & densidad_verde[niveles[2]], densidad_resultante['muy_bajo']))

    # niveles = ['muy_muy_bajo', 'muy_bajo', 'bajo', 'medio', 'alto', 'muy_alto', 'muy_muy_alto']
    # Para densidad baja
    rules.append(ctrl.Rule(densidad_verde[niveles[1]] & densidad_rojo[niveles[2]], densidad_resultante['bajo']))
    rules.append(ctrl.Rule(densidad_verde[niveles[1]] & densidad_rojo[niveles[3]], densidad_resultante['bajo']))
    rules.append(ctrl.Rule(densidad_verde[niveles[2]] & densidad_rojo[niveles[3]], densidad_resultante['bajo']))
    rules.append(ctrl.Rule(densidad_verde[niveles[2]] & densidad_rojo[niveles[4]], densidad_resultante['bajo']))
    rules.append(ctrl.Rule(densidad_verde[niveles[3]] & densidad_rojo[niveles[4]], densidad_resultante['bajo']))
    rules.append(ctrl.Rule(densidad_verde[niveles[3]] & densidad_rojo[niveles[5]], densidad_resultante['bajo']))
    rules.append(ctrl.Rule(densidad_verde[niveles[4]] & densidad_rojo[niveles[5]], densidad_resultante['bajo']))

    rules.append(ctrl.Rule(densidad_verde[niveles[0]] & densidad_rojo[niveles[1]], densidad_resultante['bajo']))
    rules.append(ctrl.Rule(densidad_verde[niveles[0]] & densidad_rojo[niveles[2]], densidad_resultante['bajo']))

    rules.append(ctrl.Rule(densidad_verde[niveles[4]] & densidad_rojo[niveles[6]], densidad_resultante['bajo']))
    rules.append(ctrl.Rule(densidad_verde[niveles[5]] & densidad_rojo[niveles[6]], densidad_resultante['bajo']))

    # Crear el sistema de control difuso
    traffic_ctrl = ctrl.ControlSystem(rules)

    # Crear la simulación
    traffic = ctrl.ControlSystemSimulation(traffic_ctrl)

    # Ahora puede usar traffic.input para establecer los valores de entrada
    # y traffic.compute() para calcular la densidad resultante

    return traffic



# Función para actualizar las funciones de pertenencia de la lógica difusa

def update_membership_functions(prediccion_rojo, prediccion_verde):
    # Definir las funciones de membresía para densidad_rojo y densidad_verde
    niveles = ['muy_muy_bajo', 'muy_bajo', 'bajo', 'medio', 'alto', 'muy_alto', 'muy_muy_alto']

    for var in densidad_rojo:
        # Define 7 equally distributed fuzzy membership functions

        n_classes = 5  # The number of classes without the first and last one
        mid_point = prediccion_rojo  # The central point for distribution

        low_setpoint = mid_point / (n_classes - 2)  # Adjusted for central point

        high_setpoint = (1 - mid_point) / (n_classes - 2)  # Adjusted for central point

        for i in range(n_classes - 2):
            start = max(0, (i - 1) * low_setpoint)
            end = min((i + 1) * low_setpoint, 1)
            if i == 0:  # For the first membership function
                fuzz.append(fuzz.trimf(var.universe, [0, 0, end]))
            else:  # For the intermediate membership functions
                fuzz.append(fuzz.trimf(var.universe, [start, i * low_setpoint, end]))

        for i in range(n_classes - 1):
            start = mid_point + max(0, (i - 1) * high_setpoint)
            end = mid_point + min((i + 1) * high_setpoint, 1)
            if i == 0:
                start = mid_point - low_setpoint
                fuzz.append(fuzz.trimf(var.universe, [start, mid_point + i * high_setpoint, end]))
            elif i == n_classes + 1:  # For the last membership function
                fuzz.append(fuzz.trimf(var.universe, [start, 1, 1]))
            else:  # For the intermediate membership functions
                fuzz.append(fuzz.trimf(var.universe, [start, mid_point + i * high_setpoint, end]))

    for var in densidad_verde:

        # Define 7 equally distributed fuzzy membership functions

        n_classes = 5  # The number of classes without the first and last one
        mid_point = prediccion_verde  # The central point for distribution

        low_setpoint = mid_point / (n_classes - 2)  # Adjusted for central point

        high_setpoint = (1 - mid_point) / (n_classes - 2)  # Adjusted for central point

        for i in range(n_classes - 2):
            start = max(0, (i - 1) * low_setpoint)
            end = min((i + 1) * low_setpoint, 1)
            if i == 0:  # For the first membership function
                fuzz.append(fuzz.trimf(var.universe, [0, 0, end]))
            else:  # For the intermediate membership functions
                fuzz.append(fuzz.trimf(var.universe, [start, i * low_setpoint, end]))

        for i in range(n_classes - 1):
            start = mid_point + max(0, (i - 1) * high_setpoint)
            end = mid_point + min((i + 1) * high_setpoint, 1)
            if i == 0:
                start = mid_point - low_setpoint
                fuzz.append(fuzz.trimf(var.universe, [start, mid_point + i * high_setpoint, end]))
            elif i == n_classes + 1:  # For the last membership function
                fuzz.append(fuzz.trimf(var.universe, [start, 1, 1]))
            else:  # For the intermediate membership functions
                fuzz.append(fuzz.trimf(var.universe, [start, mid_point + i * high_setpoint, end]))

    # Actualizar las reglas de la lógica difusa en el controlador'
    # Definir las funciones de membresía para densidad_resultante

    densidad_resultante['muy_muy_bajo'] = fuzz.trimf(densidad_resultante.universe, [0, 0, 0.167])
    densidad_resultante['muy_bajo'] = fuzz.trimf(densidad_resultante.universe, [0, 0.167, 0.334])
    densidad_resultante['bajo'] = fuzz.trimf(densidad_resultante.universe, [0.167, 0.334, 0.501])
    densidad_resultante['medio'] = fuzz.trimf(densidad_resultante.universe, [0.334, 0.501, 0.668])
    densidad_resultante['alto'] = fuzz.trimf(densidad_resultante.universe, [0.501, 0.668, 0.835])
    densidad_resultante['muy_alto'] = fuzz.trimf(densidad_resultante.universe, [0.668, 0.835, 1])
    densidad_resultante['muy_muy_alto'] = fuzz.trimf(densidad_resultante.universe, [0.835, 1, 1])

    # La actualización se puede hacer de acuerdo a la nueva lógica y reglas.
    # Nota: Aquí puedes actualizar las reglas en el controlador que se ha creado en la función defuzzify.
    # Definir las reglas de la lógica difusa

    rules = []

    # Para densidad media
    rules.append(ctrl.Rule(densidad_rojo[niveles[0]] & densidad_verde[niveles[0]], densidad_resultante['medio']))
    rules.append(ctrl.Rule(densidad_rojo[niveles[1]] & densidad_verde[niveles[1]], densidad_resultante['medio']))
    rules.append(ctrl.Rule(densidad_rojo[niveles[2]] & densidad_verde[niveles[2]], densidad_resultante['medio']))
    rules.append(ctrl.Rule(densidad_rojo[niveles[3]] & densidad_verde[niveles[3]], densidad_resultante['medio']))
    rules.append(ctrl.Rule(densidad_rojo[niveles[4]] & densidad_verde[niveles[4]], densidad_resultante['medio']))
    rules.append(ctrl.Rule(densidad_rojo[niveles[5]] & densidad_verde[niveles[5]], densidad_resultante['medio']))

    # Para densidad muy muy alta
    rules.append(ctrl.Rule(densidad_rojo[niveles[0]] & densidad_verde[niveles[6]], densidad_resultante['muy_muy_alto']))
    rules.append(ctrl.Rule(densidad_rojo[niveles[0]] & densidad_verde[niveles[5]], densidad_resultante['muy_muy_alto']))
    rules.append(ctrl.Rule(densidad_rojo[niveles[1]] & densidad_verde[niveles[6]], densidad_resultante['muy_muy_alto']))

    # Para densidad muy muy bajo
    rules.append(ctrl.Rule(densidad_rojo[niveles[6]] & densidad_verde[niveles[0]], densidad_resultante['muy_muy_bajo']))
    rules.append(ctrl.Rule(densidad_rojo[niveles[5]] & densidad_verde[niveles[0]], densidad_resultante['muy_muy_bajo']))
    rules.append(ctrl.Rule(densidad_rojo[niveles[6]] & densidad_verde[niveles[1]], densidad_resultante['muy_muy_bajo']))

    # Para densidad muy alta
    rules.append(ctrl.Rule(densidad_rojo[niveles[2]] & densidad_verde[niveles[5]], densidad_resultante['muy_alto']))
    rules.append(ctrl.Rule(densidad_rojo[niveles[1]] & densidad_verde[niveles[4]], densidad_resultante['muy_alto']))
    rules.append(ctrl.Rule(densidad_rojo[niveles[1]] & densidad_verde[niveles[5]], densidad_resultante['muy_alto']))

    rules.append(ctrl.Rule(densidad_rojo[niveles[0]] & densidad_verde[niveles[3]], densidad_resultante['muy_alto']))
    rules.append(ctrl.Rule(densidad_rojo[niveles[0]] & densidad_verde[niveles[4]], densidad_resultante['muy_alto']))

    rules.append(ctrl.Rule(densidad_rojo[niveles[2]] & densidad_verde[niveles[6]], densidad_resultante['muy_alto']))
    rules.append(ctrl.Rule(densidad_rojo[niveles[3]] & densidad_verde[niveles[6]], densidad_resultante['muy_alto']))

    # Para densidad alta
    rules.append(ctrl.Rule(densidad_rojo[niveles[1]] & densidad_verde[niveles[2]], densidad_resultante['alto']))
    rules.append(ctrl.Rule(densidad_rojo[niveles[1]] & densidad_verde[niveles[3]], densidad_resultante['alto']))
    rules.append(ctrl.Rule(densidad_rojo[niveles[2]] & densidad_verde[niveles[3]], densidad_resultante['alto']))
    rules.append(ctrl.Rule(densidad_rojo[niveles[2]] & densidad_verde[niveles[4]], densidad_resultante['alto']))
    rules.append(ctrl.Rule(densidad_rojo[niveles[3]] & densidad_verde[niveles[4]], densidad_resultante['alto']))
    rules.append(ctrl.Rule(densidad_rojo[niveles[3]] & densidad_verde[niveles[5]], densidad_resultante['alto']))
    rules.append(ctrl.Rule(densidad_rojo[niveles[4]] & densidad_verde[niveles[5]], densidad_resultante['alto']))

    rules.append(ctrl.Rule(densidad_rojo[niveles[0]] & densidad_verde[niveles[1]], densidad_resultante['alto']))
    rules.append(ctrl.Rule(densidad_rojo[niveles[0]] & densidad_verde[niveles[2]], densidad_resultante['alto']))

    rules.append(ctrl.Rule(densidad_rojo[niveles[4]] & densidad_verde[niveles[6]], densidad_resultante['alto']))
    rules.append(ctrl.Rule(densidad_rojo[niveles[5]] & densidad_verde[niveles[6]], densidad_resultante['alto']))

    # Para densidad muy baja
    rules.append(ctrl.Rule(densidad_rojo[niveles[5]] & densidad_verde[niveles[1]], densidad_resultante['muy_bajo']))
    rules.append(ctrl.Rule(densidad_rojo[niveles[4]] & densidad_verde[niveles[1]], densidad_resultante['muy_bajo']))
    rules.append(ctrl.Rule(densidad_rojo[niveles[5]] & densidad_verde[niveles[2]], densidad_resultante['muy_bajo']))

    rules.append(ctrl.Rule(densidad_rojo[niveles[4]] & densidad_verde[niveles[0]], densidad_resultante['muy_bajo']))
    rules.append(ctrl.Rule(densidad_rojo[niveles[3]] & densidad_verde[niveles[0]], densidad_resultante['muy_bajo']))

    rules.append(ctrl.Rule(densidad_rojo[niveles[6]] & densidad_verde[niveles[3]], densidad_resultante['muy_bajo']))
    rules.append(ctrl.Rule(densidad_rojo[niveles[6]] & densidad_verde[niveles[2]], densidad_resultante['muy_bajo']))

    # niveles = ['muy_muy_bajo', 'muy_bajo', 'bajo', 'medio', 'alto', 'muy_alto', 'muy_muy_alto']
    # Para densidad baja
    rules.append(ctrl.Rule(densidad_verde[niveles[1]] & densidad_rojo[niveles[2]], densidad_resultante['bajo']))
    rules.append(ctrl.Rule(densidad_verde[niveles[1]] & densidad_rojo[niveles[3]], densidad_resultante['bajo']))
    rules.append(ctrl.Rule(densidad_verde[niveles[2]] & densidad_rojo[niveles[3]], densidad_resultante['bajo']))
    rules.append(ctrl.Rule(densidad_verde[niveles[2]] & densidad_rojo[niveles[4]], densidad_resultante['bajo']))
    rules.append(ctrl.Rule(densidad_verde[niveles[3]] & densidad_rojo[niveles[4]], densidad_resultante['bajo']))
    rules.append(ctrl.Rule(densidad_verde[niveles[3]] & densidad_rojo[niveles[5]], densidad_resultante['bajo']))
    rules.append(ctrl.Rule(densidad_verde[niveles[4]] & densidad_rojo[niveles[5]], densidad_resultante['bajo']))

    rules.append(ctrl.Rule(densidad_verde[niveles[0]] & densidad_rojo[niveles[1]], densidad_resultante['bajo']))
    rules.append(ctrl.Rule(densidad_verde[niveles[0]] & densidad_rojo[niveles[2]], densidad_resultante['bajo']))

    rules.append(ctrl.Rule(densidad_verde[niveles[4]] & densidad_rojo[niveles[6]], densidad_resultante['bajo']))
    rules.append(ctrl.Rule(densidad_verde[niveles[5]] & densidad_rojo[niveles[6]], densidad_resultante['bajo']))

    # Crear el sistema de control difuso
    traffic_ctrl = ctrl.ControlSystem(rules)

    # Crear la simulación
    traffic = ctrl.ControlSystemSimulation(traffic_ctrl)

    # Ahora puede usar traffic.input para establecer los valores de entrada
    # y traffic.compute() para calcular la densidad resultante

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


def get_edge_id_from_lane_id(laneID):
    edgeID = laneID.split('_')[0]
    return edgeID


# contains TraCI control loop
def get_density(interseccion_id, calle):
    num = traci.edge.getLastStepVehicleNumber(get_edge_id_from_lane_id(calle))
    largo = traci.lane.getLength(calle)
    density = (num / traci.lane.getLength(calle))
    # print(f" calle  numero  : {calle} con densidad  {density} una cantidad de carros de {num} y un largo de {largo}")

    return density

def get_queuelngth(calle):
    # Inicializa un contador para la longitud de la cola.
    queue_length = 0.0

    # ID del carril
    lane_id = calle

    # Obtiene los IDs de los vehículos en el carril.
    vehicles_on_lane = traci.lane.getLastStepVehicleIDs(lane_id)
    largo_calle=traci.lane.getLength(calle)

    # Para cada vehículo en el carril, comprueba si su velocidad está por debajo de un cierto umbral.
    # Si es así, consideramos que el vehículo está en la cola y agregamos su longitud a queue_length.
    for vehicle_id in vehicles_on_lane:
        if traci.vehicle.getSpeed(vehicle_id) < 0.1: # velocidad menor a 0.1 m/s
            queue_length += traci.vehicle.getLength(vehicle_id)

    largo_cola_norm= queue_length/largo_calle
    return largo_cola_norm

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
    controladores =[]
    calles = []
    # un ciclo que recorra todas las intersecciones
    models_R =[]
    models_V = []
    steps = 0
    # Modificar la sección del bucle principal
    densidades_acumuladas = []
    densidades_acumuladas_rojo = []
    densidades_acumuladas_verde = []

    # Acumular largos de cola en cada paso
    largo_cola_acumuladas = []
    largo_cola_verde_acumuladas = []
    largo_cola_rojo_acumuladas = []
    num_time_steps = 10
    num_features = 2

    model = tf.keras.Sequential([
        tf.keras.layers.Conv1D(32, 3, activation='relu', input_shape=(num_time_steps, num_features)),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Conv1D(64, 3, activation='relu'),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
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
        models_R.append(model)
        models_V.append(model)
        controladores.append(controlador)


    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        steps = steps + 1

        # Acumular densidades en cada paso
        densidades = []
        densidades_verde = []
        densidades_rojo = []



        for i in range(len(intersecciones)):
            if interseccion_id != 230:
                interseccion_id = intersecciones[i]
                ci = calles[i]
                # Obtener el estado del semáforo
                estado_semaforo = traci.trafficlight.getRedYellowGreenState(interseccion_id)
                densidades_acumuladas_rojo.append([])
                densidades_acumuladas_verde.append([])
                for c in ci:
                    densidad = get_density(interseccion_id, c)
                    # if densidad > 0:
                    densidades.append(densidad)

                    estado_calle = estado_semaforo[ci.index(c)]

                    # Agregar la densidad a la lista correspondiente según el estado del semáforo
                    if estado_calle == 'G':
                        densidades_verde.append(densidad)
                    elif estado_calle == 'r':
                        densidades_rojo.append(densidad)
                densidades_acumuladas_rojo[i].append(np.mean(densidades_rojo))
                densidades_acumuladas_verde[i].append(np.mean(densidades_verde))
        if densidades:
            densidades_acumuladas.append(np.mean(densidades))

        # Acumular largos de cola en cada paso
        largo_cola = []
        largo_cola_verde = []
        largo_cola_rojo = []

        for i in range(len(intersecciones)):
            if interseccion_id != 230:
                interseccion_id = intersecciones[i]
                ci = calles[i]
                # Obtener el estado del semáforo
                estado_semaforo = traci.trafficlight.getRedYellowGreenState(interseccion_id)
                largo_cola_rojo_acumuladas.append([])
                largo_cola_verde_acumuladas.append([])
                for c in ci:

                    largo=get_queuelngth(c)
                    # if densidad > 0:
                    largo_cola.append(largo)

                    estado_calle = estado_semaforo[ci.index(c)]

                    # Agregar la densidad a la lista correspondiente según el estado del semáforo
                    if estado_calle == 'G':
                        largo_cola_verde.append(largo)
                    elif estado_calle == 'r':
                        largo_cola_rojo.append(largo)
                largo_cola_rojo_acumuladas[i].append(np.mean(largo_cola_rojo))
                largo_cola_verde_acumuladas[i].append(np.mean(largo_cola_verde))
        if largo_cola:
            largo_cola_acumuladas.append(np.mean(largo_cola))




        if steps % 300 == 0:
            # if steps < 0:

            # Continuar con el resto de la lógica para el control de semáforos

            # print(len(intersecciones))
            for i in range(len(intersecciones)):
                interseccion_id = intersecciones[i]

                s = interseccion_id

                if steps % 3000 == 0:
                    # if steps < 0:
                    print("Conteo de pasos: ", steps)
                    # Preparar los datos para el entrenamiento de la red neuronal



                    X_train, y_train = create_windowed_data(largo_cola_rojo_acumuladas[:-1], num_time_steps)

                    # Primero calculamos cuántos elementos necesitamos eliminar
                    num_elements = X_train.size
                    num_elements_to_remove = num_elements % (num_time_steps * num_features)

                    # Luego eliminamos los elementos sobrantes del final del array
                    # Usamos np.resize para cambiar el tamaño de X_train
                    if num_elements_to_remove > 0:
                        X_train = np.resize(X_train, num_elements - num_elements_to_remove)

                    # Ahora puedes redimensionar X_train con éxito


                    X_train = X_train.reshape(-1, num_time_steps, num_features)
                    y_train = y_train.reshape(-1, 1)

                    # Entrenar la red neuronal LSTM
                    tf.config.run_functions_eagerly(True)
                    models_R[i].compile(loss='mean_squared_error', optimizer='adam', run_eagerly=True)
                    models_R[i].fit(X_train, y_train, epochs=100, batch_size=64)

                    # Predecir el próximo valor de densidad

                    nuevo_largo_rojo = models_R[i].predict(X_train[-1].reshape(1, num_time_steps, num_features))[0][0]

                    X_train, y_train = create_windowed_data(largo_cola_verde_acumuladas[:-1], num_time_steps)

                    # Primero calculamos cuántos elementos necesitamos eliminar
                    num_elements = X_train.size
                    num_elements_to_remove = num_elements % (num_time_steps * num_features)

                    # Luego eliminamos los elementos sobrantes del final del array
                    # Usamos np.resize para cambiar el tamaño de X_train
                    if num_elements_to_remove > 0:
                        X_train = np.resize(X_train, num_elements - num_elements_to_remove)

                    # Ahora puedes redimensionar X_train con éxito
                    X_train = X_train.reshape(-1, num_time_steps, num_features)

                    X_train = X_train.reshape(-1, num_time_steps, num_features)
                    y_train = y_train.reshape(-1, 1)

                    # Entrenar la red neuronal LSTM
                    tf.config.run_functions_eagerly(True)
                    models_V[i].compile(loss='mean_squared_error', optimizer='adam', run_eagerly=True)
                    models_V[i].fit(X_train, y_train, epochs=100, batch_size=64)

                    # Predecir el próximo valor de densidad

                    nuevo_largo_verde = models_V[i].predict(X_train[-1].reshape(1, num_time_steps, num_features))[0][0]



                    controladores[i] = update_membership_functions(nuevo_largo_rojo, nuevo_largo_verde)

                # Obtener la definición del semáforo

                definicion_semaforos = traci.trafficlight.getAllProgramLogics(s)

                indice_fase = traci.trafficlight.getPhase(s)
                num_fases = len(definicion_semaforos[0].getPhases())

                estado_semaforo = traci.trafficlight.getRedYellowGreenState(s)
                # print(f"Intersección  numero  : {s} con densidad promedio en verde de {densidades_verde[i]} una densidad promedio en rojo de {densidades_rojo[i]} ")

                # En su lógica difusa, utilice la densidad promedio de las calles en rojo y verde en lugar de la densidad por calle
                largo_cola_verde_promedio = np.mean(largo_cola_verde[i]) if densidades_verde[i] else 0
                largo_cola_rojo_promedio = np.mean(largo_cola_rojo[i]) if densidades_rojo[i] else 0

                controladores[i].input['rojo'] = largo_cola_rojo_promedio
                controladores[i].input['verde'] = largo_cola_verde_promedio

                controladores[i].compute()

                if (controladores[i].output['resultante'] <= 0.5):
                    # if (desicion_ajustada >= 0.5):

                    traci.trafficlight.setPhase(s, indice_fase)

                    print(
                        f"Intersección gestionada numero  : {s} con grado de pertenencia {controlador.output['resultante']}")
                    # break

    # Desconexión de SUMO

    traci.close()


def start_progressbar():
    options = get_options()

    # check binary
    if options.nogui:
        sumoBinary = checkBinary('sumo')
    else:
        sumoBinary = checkBinary('sumo-gui')
    for i in range(3):
        seed = random.randint(1, 10000)  # Genera una semilla aleatoria entre 1 y 10000
        queue_output_file = f"colaControl{seed}.xml"  # Modifica el nombre del archivo de salida para incluir la semilla

        traci.start([sumoBinary, "-c", "run.sumocfg",
                     "--tripinfo-output", "tripinfo.xml", "--queue-output", queue_output_file, "--seed", str(seed)])
        run()

        # Incrementa la barra de progreso
        progressbar['value'] += 1
        window.update_idletasks()


# main entry point
if __name__ == "__main__":
    window = tk.Tk()
    window.title('Barra de progreso')

    progressbar = ttk.Progressbar(window, length=300, mode='determinate')
    progressbar.pack(pady=10)
    progressbar['maximum'] = 3
    progressbar['value'] = 0

    # Correr la simulación en un hilo separado para no bloquear la interfaz gráfica
    thread = threading.Thread(target=start_progressbar)
    thread.start()

    window.mainloop()
