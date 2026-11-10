#!/usr/bin/env python
import os
import sys
import optparse
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import tensorflow as tf
import random
import tkinter as tk
from tkinter import ttk
import threading
from sumolib import checkBinary  # Checks for the binary in environ vars
import traci
import csv
tf.data.experimental.enable_debug_mode()

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

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
    if prediccion_rojo<0:
        prediccion_rojo=0
    if prediccion_verde<0:
        prediccion_verde=0
    mfR = []
    print("predicciÓN DE rojo",prediccion_rojo )
    print("predicciÓN DE verde", prediccion_verde)
    for var in [densidad_rojo]:
        # Define 7 equally distributed fuzzy membership functions

        n_classes = 5  # The number of classes without the first and last one
        mid_point = prediccion_rojo  # The central point for distribution

        low_setpoint = mid_point / (n_classes - 2)  # Adjusted for central point

        high_setpoint = (1 - mid_point) / (n_classes - 2)  # Adjusted for central point
        if low_setpoint<0: low_setpoint=0
        if high_setpoint < 0: high_setpoint = 0
        for i in range(n_classes - 2):
            start = max(0, (i - 1) * low_setpoint)
            end = min((i + 1) * low_setpoint, 1)

            if i == 0:  # For the first membership function
                mfR.append(fuzz.trimf(var.universe, [0, 0, end]))
            else:  # For the intermediate membership functions
                mfR.append(fuzz.trimf(var.universe, [start, i * low_setpoint, end]))

        for i in range(n_classes - 1):
            start = mid_point + max(0, (i - 1) * high_setpoint)
            end = mid_point + min((i + 1) * high_setpoint, 1)
            if i == 0:
                start = mid_point - low_setpoint
                mfR.append(fuzz.trimf(var.universe, [start, mid_point + i * high_setpoint, end]))
            elif i == n_classes + 1:  # For the last membership function
                mfR.append(fuzz.trimf(var.universe, [start, 1, 1]))
            else:  # For the intermediate membership functions
                mfR.append(fuzz.trimf(var.universe, [start, mid_point + i * high_setpoint, end]))
    mfV = []
    for var in [densidad_verde]:

        # Define 7 equally distributed fuzzy membership functions

        n_classes = 5  # The number of classes without the first and last one
        mid_point = prediccion_verde  # The central point for distribution

        low_setpoint = mid_point / (n_classes - 2)  # Adjusted for central point

        high_setpoint = (1 - mid_point) / (n_classes - 2)  # Adjusted for central point
        if low_setpoint<0: low_setpoint=0
        if high_setpoint < 0: high_setpoint = 0
        for i in range(n_classes - 2):
            start = max(0, (i - 1) * low_setpoint)
            end = min((i + 1) * low_setpoint, 1)
            if i == 0:  # For the first membership function
                mfV.append(fuzz.trimf(var.universe, [0, 0, end]))
            else:  # For the intermediate membership functions
                mfV.append(fuzz.trimf(var.universe, [start, i * low_setpoint, end]))

        for i in range(n_classes - 1):
            start = mid_point + max(0, (i - 1) * high_setpoint)
            end = mid_point + min((i + 1) * high_setpoint, 1)
            if i == 0:
                start = mid_point - low_setpoint
                mfV.append(fuzz.trimf(var.universe, [start, mid_point + i * high_setpoint, end]))
            elif i == n_classes + 1:  # For the last membership function
                mfV.append(fuzz.trimf(var.universe, [start, 1, 1]))
            else:  # For the intermediate membership functions
                mfV.append(fuzz.trimf(var.universe, [start, mid_point + i * high_setpoint, end]))

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
    print("Llegamos hasta aca")

    densidad_rojo[niveles[0]]=mfR[0]
    densidad_rojo[niveles[1]]=mfR[1]
    densidad_rojo[niveles[2]]=mfR[2]
    densidad_rojo[niveles[3]]=mfR[3]
    densidad_rojo[niveles[4]]=mfR[4]
    densidad_rojo[niveles[5]]=mfR[5]

    densidad_verde[niveles[0]]=mfV[0]
    densidad_verde[niveles[1]]=mfV[1]
    densidad_verde[niveles[2]]=mfV[2]
    densidad_verde[niveles[3]]=mfV[3]
    densidad_verde[niveles[4]]=mfV[4]
    densidad_verde[niveles[5]]=mfV[5]

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
    print("Terminamos las reglas")
    # Crear el sistema de control difuso
    traffic_ctrl = ctrl.ControlSystem(rules)

    # Crear la simulación
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


def get_options2():
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


def get_queuelngth(calle):
    # Inicializa un contador para la longitud de la cola.
    queue_length = 0.0

    # ID del carril
    lane_id = calle

    # Obtiene los IDs de los vehículos en el carril.
    vehicles_on_lane = traci.lane.getLastStepVehicleIDs(lane_id)
    largo_calle = traci.lane.getLength(calle)

    # Para cada vehículo en el carril, comprueba si su velocidad está por debajo de un cierto umbral.
    # Si es así, consideramos que el vehículo está en la cola y agregamos su longitud a queue_length.
    for vehicle_id in vehicles_on_lane:
        if traci.vehicle.getSpeed(vehicle_id) < 0.1:  # velocidad menor a 0.1 m/s
            queue_length += traci.vehicle.getLength(vehicle_id)

    largo_cola_norm = queue_length / largo_calle
    return largo_cola_norm


#def create_windowed_data(data, window_size):
#    X = []
#    y = []

#    for i in range(len(data) - window_size):
#        X.append(data[i:i + window_size])
#        y.append(data[i + window_size])

#    return np.array(X), np.array(y)

def create_sequences(data, seq_length):
    xs = []
    ys = []

    for i in range(len(data)-seq_length-1):
        x = data[i:(i+seq_length)]
        y = data[i+seq_length]
        #print (data)
        #print (x)
        #ptint (y)
        xs.append(np.array(x))  # convierte a np.array aquí
        ys.append(np.array(y))  # convierte a np.array aquí

    return np.array(xs, "float32"), np.array(ys, "float32")  # asegúrate de que sean ndarrays


def run():
    # Obtener la lista de ID de intersecciones
    intersecciones = traci.trafficlight.getIDList()
    largo_cola_rojo_acumuladas=[]
    largo_cola_verde_acumuladas=[]

    controlador = defuzzify()
    controladores = [controlador for _ in intersecciones]

    num_time_steps = 1
    num_features = 1

    # Crear un único modelo LSTM para ser utilizado en todas las intersecciones
    model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(50, activation='relu', input_shape=(num_time_steps, num_features)),
        tf.keras.layers.Dense(1)
    ])
    model.compile(loss='mean_squared_error', optimizer='adam', run_eagerly=True)
    modelsR = [model for _ in intersecciones]  # Copia el modelo para cada intersección
    modelsV = [model for _ in intersecciones]  # Copia el modelo para cada intersección

    steps = 0
    largo_cola_acumuladas = []

    # Bucle principal
    start_index = 0
    end_index = start_index + 600

    # Aquí almacenaremos todas las lanes únicas con semáforos
    lanes_con_semaforos = set()

    for i, interseccion_id in enumerate(intersecciones):
        conexiones = traci.trafficlight.getControlledLanes(interseccion_id)
        lanes_con_semaforos.update(conexiones)  # agregar a nuestro conjunto, eliminando duplicados
    # Ahora escribimos las lanes con semáforos en un archivo csv
    with open('lanes_con_semaforos.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Lane'])  # encabezado de la columna
        for lane in lanes_con_semaforos:
            writer.writerow([lane])

    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        steps += 1

        largo_cola = []
        largo_cola_verde = []
        largo_cola_rojo = []

        for i, interseccion_id in enumerate(intersecciones):
            conexiones = traci.trafficlight.getControlledLanes(interseccion_id)
            ci = list(set(conexiones))  # Eliminar duplicados
            largo_cola_rojo_acumuladas.append([])
            largo_cola_verde_acumuladas.append([])
            # Calcula la longitud de la cola para cada calle controlada
            for c in ci:
                largo = get_queuelngth(c)
                if (largo):
                    largo_cola.append(largo)
                else:
                    largo=0
                    largo_cola.append(largo)

                # Agrega el largo a la lista correspondiente según el estado del semáforo

                estado_calle = traci.trafficlight.getRedYellowGreenState(interseccion_id)[ci.index(c)]
                if estado_calle == 'G':
                    largo_cola_verde.append(largo)
                elif estado_calle == 'r':
                    largo_cola_rojo.append(largo)

            # Actualiza las listas acumuladas
            largo_cola_rojo_acumuladas[i].append(np.mean(largo_cola_rojo) if largo_cola_rojo else 0)
            largo_cola_verde_acumuladas[i].append(np.mean(largo_cola_verde) if largo_cola_verde else 0)
            #print ("largo rojo",largo_cola_rojo_acumuladas[i])
        if largo_cola:
            largo_cola_acumuladas.append(np.mean(largo_cola))

        if steps % 300 == 0:
            # Actualiza el modelo LSTM para cada intersección
            for i, interseccion_id in enumerate(intersecciones):
                if steps % 3000 == 0:
                    #print(f"Conteo de pasos: {steps}")
                    # Entrena y utiliza el modelo LSTM

                    tf.config.run_functions_eagerly(True)
                    #print("rojo acumulada", len(largo_cola_rojo_acumuladas))
                    #print("verde acumulada", len(largo_cola_verde_acumuladas))
                    #for largo_cola_acumulada, modelo in zip([largo_cola_rojo_acumuladas, largo_cola_verde_acumuladas], models):

                    X_train, y_train = create_sequences(largo_cola_rojo_acumuladas[i],
                                                        num_time_steps)


                    #X_train = X_train.reshape(-1, num_time_steps, num_features)
                    #y_train = y_train.reshape(-1, 1)

                    # Para comprobar NaNs o infinitos
                    assert not np.isnan(X_train).any(), "X_train tiene NaNs"
                    assert not np.isnan(y_train).any(), "y_train tiene NaNs"
                    assert not np.isinf(X_train).any(), "X_train tiene valores infinitos"
                    assert not np.isinf(y_train).any(), "y_train tiene valores infinitos"

                    #modelsR[i].compile(loss='mean_squared_error', optimizer='adam', run_eagerly=True)
                    #print("equis", X_train)
                    #print("I griega", y_train)
                    modelsR[i].fit(X_train, y_train, epochs=5, batch_size=20)
                    # Actualiza las funciones de pertenencia del controlador difuso
                    nuevo_largo = modelsR[i].predict(X_train[-1].reshape(1, num_time_steps, num_features))[0][0]

                    X_train, y_train = create_sequences(largo_cola_verde_acumuladas[i],
                                                        num_time_steps)

                    #X_train = X_train.reshape(-1, num_time_steps, num_features)
                    #y_train = y_train.reshape(-1, 1)
                    modelsV[i].fit(X_train, y_train, epochs=5, batch_size=20)
                    # Actualiza las funciones de pertenencia del controlador difuso
                    nuevo_largoV = modelsV[i].predict(X_train[-1].reshape(1, num_time_steps, num_features))[0][0]

                    controladores[i] = update_membership_functions(nuevo_largo, nuevo_largoV)

                    #start_index = end_index
                    #end_index = start_index + 600

                # Control difuso
                largo_cola_verde_promedio = np.mean(largo_cola_verde[i]) if largo_cola_verde[i] else 0
                largo_cola_rojo_promedio = np.mean(largo_cola_rojo[i]) if largo_cola_rojo[i] else 0
                controladores[i].input['rojo'] = largo_cola_rojo_promedio
                controladores[i].input['verde'] = largo_cola_verde_promedio
                try:
                    controladores[i].compute()
                except:
                    print("No se pudo gestionar")
                largo_cola_rojo_acumuladas[i] = []
                largo_cola_verde_acumuladas[i] = []
                if controladores[i].output['resultante'] <= 0.5:
                    traci.trafficlight.setPhase(interseccion_id, traci.trafficlight.getPhase(interseccion_id))
                    print(f"Calle gestionada numero  : {interseccion_id} con largo de cola {controladores[i].output['resultante']}")

    # Cierra la conexión con SUMO
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
