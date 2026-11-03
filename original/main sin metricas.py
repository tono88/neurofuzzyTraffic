#!/usr/bin/env python
"""
Multi-mode traffic control para SUMO: NeuroFuzzy, Fuzzy y Estándar.

Este script ejecuta tres estrategias de control:
 - NeuroFuzzy: control difuso con predicción vía MLP.
 - Fuzzy: control sólo difuso.
 - Estándar: sin intervención en semáforos.

Para cada estrategia se realizan 100 corridas, guardando los archivos de salida en carpetas:
  • NeuroFuzzy/
  • Fuzzy/
  • Estandar/

Requisitos:
  • SUMO_HOME apuntando al directorio de SUMO.
  • run.sumocfg presente en el directorio de ejecución.
"""
import os
import sys
import optparse
import logging
from collections import deque

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import tensorflow as tf

# --- Configuración de logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger(__name__)

# Ejecutar TF en modo ansioso para debugging
tf.config.run_functions_eagerly(True)
logger.info("Num GPUs Available: %d", len(tf.config.list_physical_devices('GPU')))

# --- Inicialización SUMO / TraCI ---
def init_sumo():
    if 'SUMO_HOME' not in os.environ:
        sys.exit("ERROR: Declare la variable de entorno SUMO_HOME apuntando a tu instalación de SUMO.")
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
    from sumolib import checkBinary
    import traci
    return checkBinary, traci

checkBinary, traci = init_sumo()

# --- Opciones de línea de comando ---
def get_options():
    parser = optparse.OptionParser()
    parser.add_option(
        "--nogui", action="store_true", default=True,
        help="Ejecutar SUMO en modo consola (sin GUI)"
    )
    opts, _ = parser.parse_args()
    return opts

# --- Función para obtener densidad en escala 0-100 ---
def get_density(lane_id):
    length = traci.lane.getLength(lane_id)
    if length <= 0:
        return 0.0
    count = len(traci.lane.getLastStepVehicleIDs(lane_id))
    return (count / length) * 100.0

# --- Definición del sistema difuso ---
d_man = np.arange(0, 101, 1)
dout  = np.arange(0, 1.01, 0.01)

densidad_vehicular = ctrl.Antecedent(d_man, 'densidad')
cambio_estado      = ctrl.Consequent(dout, 'estado')

def defuzzify():
    # Membresías estáticas
    densidad_vehicular['low']    = fuzz.trapmf(d_man, [0,   0,  30,  50])
    densidad_vehicular['medium'] = fuzz.trapmf(d_man, [30,  50, 70, 100])
    densidad_vehicular['high']   = fuzz.trapmf(d_man, [70, 100,100,100])

    cambio_estado['actual']    = fuzz.trimf(dout, [0.0, 0.0, 0.5])
    cambio_estado['siguiente'] = fuzz.trimf(dout, [0.5, 1.0, 1.0])

    r1 = ctrl.Rule(densidad_vehicular['low'], cambio_estado['actual'])
    r2 = ctrl.Rule(
        densidad_vehicular['medium'] | densidad_vehicular['high'],
        cambio_estado['siguiente']
    )
    system = ctrl.ControlSystem([r1, r2])
    return ctrl.ControlSystemSimulation(system)

# --- Ajuste dinámico de membresías vía predicción ---
def update_membership_functions(pred):
    # Clip para garantizar pred ∈ [0,100]
    pred_clipped = float(np.clip(pred, 0, 100))
    # Definir puntos relativos
    low_ub  = pred_clipped * 0.5
    med_lb  = pred_clipped * 0.25
    med_ub  = pred_clipped * 0.75
    high_lb = low_ub

    # Asegurar orden a <= b <= c <= d
    densidad_vehicular['low']    = fuzz.trapmf(d_man,    [0,       0,       low_ub,   pred_clipped])
    densidad_vehicular['medium'] = fuzz.trapmf(d_man,    [med_lb,  med_ub,  pred_clipped, 100])
    densidad_vehicular['high']   = fuzz.trapmf(d_man,    [high_lb, 100,     100,        100])

    r1 = ctrl.Rule(densidad_vehicular['low'], cambio_estado['actual'])
    r2 = ctrl.Rule(
        densidad_vehicular['medium'] | densidad_vehicular['high'],
        cambio_estado['siguiente']
    )
    system = ctrl.ControlSystem([r1, r2])
    return ctrl.ControlSystemSimulation(system)

# --- Preparación de datos para MLP ---
def create_windowed_data(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i : i+window_size])
        y.append(data[i+window_size])
    return np.array(X), np.array(y)

# --- Parámetros Neuro ---
WINDOW_SIZE      = 50
BUFFER_SIZE      = 200
PREDICT_INTERVAL = 300
TRAIN_EPOCHS     = 10
TRAIN_BATCH      = 16

# --- MODO 1: NeuroFuzzy ---
def run_neurofuzzy():
    logger.info('==> Ejecutando NeuroFuzzy')
    tls_list = traci.trafficlight.getIDList()
    lanes_map = { tid: list(dict.fromkeys(traci.trafficlight.getControlledLanes(tid))) for tid in tls_list }

    buffer = deque(maxlen=BUFFER_SIZE)
    model  = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(WINDOW_SIZE,)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', run_eagerly=True)

    trained = False
    step    = 0
    sim_ctrl = defuzzify()

    init_exp = traci.simulation.getMinExpectedNumber()
    logger.info('MinExpectedNumber inicial: %d', init_exp)
    if init_exp <= 0:
        logger.warning('No hay vehículos programados. Revisar run.sumocfg o rutas.')

    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        step += 1

        dens_all = []
        for lanes in lanes_map.values():
            for lane in lanes:
                d = get_density(lane)
                if d > 0:
                    dens_all.append(d)
        if dens_all:
            buffer.append(np.mean(dens_all))

        # Entrenar una vez
        if not trained and len(buffer) > WINDOW_SIZE:
            X, y = create_windowed_data(list(buffer), WINDOW_SIZE)
            if X.shape[0] > 0:
                model.fit(X, y,
                          epochs=TRAIN_EPOCHS,
                          batch_size=TRAIN_BATCH,
                          callbacks=[tf.keras.callbacks.EarlyStopping(patience=2)],
                          verbose=0)
                trained = True
                logger.info('NeuroFuzzy: modelo entrenado con %d muestras', X.shape[0])

        # Ajuste difuso periódico
        if trained and step % PREDICT_INTERVAL == 0:
            window = np.array(list(buffer)[-WINDOW_SIZE:])
            pred   = model.predict(window.reshape(1, WINDOW_SIZE))[0][0]
            sim_ctrl = update_membership_functions(pred)
            logger.info('Difuso ajustado con predicción %.2f', pred)

        # Control de semáforos
        for tid, lanes in lanes_map.items():
            cur_phase = traci.trafficlight.getPhase(tid)
            n_phases  = len(traci.trafficlight.getAllProgramLogics(tid)[0].getPhases())
            for lane in lanes:
                d = get_density(lane)
                sim_ctrl.input['densidad'] = d
                sim_ctrl.compute()
                if sim_ctrl.output['estado'] >= 0.5:
                    new_phase = (cur_phase + 1) % n_phases
                    traci.trafficlight.setPhase(tid, new_phase)
                    logger.debug('NeuroFuzzy: TLS %s -> fase %d', tid, new_phase)

    traci.close()

# --- MODO 2: Fuzzy sin NN ---
def run_fuzzy():
    logger.info('==> Ejecutando Fuzzy')
    tls_list = traci.trafficlight.getIDList()
    lanes_map = { tid: list(dict.fromkeys(traci.trafficlight.getControlledLanes(tid))) for tid in tls_list }
    sim_ctrl = defuzzify()

    init_exp = traci.simulation.getMinExpectedNumber()
    logger.info('MinExpectedNumber inicial: %d', init_exp)

    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        for tid, lanes in lanes_map.items():
            cur_phase = traci.trafficlight.getPhase(tid)
            n_phases  = len(traci.trafficlight.getAllProgramLogics(tid)[0].getPhases())
            for lane in lanes:
                d = get_density(lane)
                sim_ctrl.input['densidad'] = d
                sim_ctrl.compute()
                if sim_ctrl.output['estado'] >= 0.5:
                    new_phase = (cur_phase + 1) % n_phases
                    traci.trafficlight.setPhase(tid, new_phase)
                    logger.debug('Fuzzy: TLS %s -> fase %d', tid, new_phase)
    traci.close()

# --- MODO 3: Estándar (sin control) ---
def run_estandar():
    logger.info('==> Ejecutando Estándar')
    init_exp = traci.simulation.getMinExpectedNumber()
    logger.info('MinExpectedNumber inicial: %d', init_exp)
    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
    traci.close()

# --- Entrada principal ---
if __name__ == '__main__':
    opts    = get_options()
    binSUMO = checkBinary('sumo') if opts.nogui else checkBinary('sumo-gui')

    modos = [
        ('NeuroFuzzy', run_neurofuzzy),
        ('Fuzzy',      run_fuzzy),
        ('Estandar',   run_estandar),
    ]

    for name, funcion in modos:
        os.makedirs(name, exist_ok=True)
        for i in range(100):
            seed      = np.random.randint(1, 10000)
            out_trip  = os.path.join(name, f'tripinfo_{seed}.xml')
            out_queue = os.path.join(name, f'queue_{seed}.xml')

            traci.start([
                binSUMO,
                '-c', 'run.sumocfg',
                '--tripinfo-output', out_trip,
                '--queue-output', out_queue,
                '--seed', str(seed)
            ])
            funcion()
