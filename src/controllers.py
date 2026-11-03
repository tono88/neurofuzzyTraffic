"""Controladores derivados de original/main.py.

Las reglas, arquitectura, orden de carriles y cambios de fase se conservan.
Las diferencias de ejecución están descritas en docs/METODOLOGIA.md.
Este módulo se importa después de configurar las semillas en el proceso hijo.
"""

from collections import deque
import sys
import logging
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import tensorflow as tf
import traci

tf.data.experimental.enable_debug_mode()
tf.get_logger().setLevel(logging.ERROR)
tf.config.run_functions_eagerly(True)
d_man = np.arange(0, 101, 1)
dout = np.arange(0, 1.01, 0.01)
initial_med_lb = 30.0
initial_med_ub = 70.0
WINDOW_SIZE = 50
BUFFER_SIZE = 200
PREDICT_INTERVAL = 300
TRAIN_EPOCHS = 2
TRAIN_BATCH = 1


def get_density(lane_id):
    """Vehículos por 100 metros de carril; no es un porcentaje de ocupación."""
    length = traci.lane.getLength(lane_id)
    return (
        len(traci.lane.getLastStepVehicleIDs(lane_id)) / length * 100.0
        if length > 0
        else 0.0
    )


def defuzzify_static():
    """Construye las dos reglas Mamdani y los conjuntos estáticos de main.py."""
    ant = ctrl.Antecedent(d_man, "densidad")
    con = ctrl.Consequent(dout, "estado")
    ant["low"] = fuzz.trapmf(d_man, [0, 0, 30, 50])
    ant["medium"] = fuzz.trapmf(d_man, [30, 50, 70, 100])
    ant["high"] = fuzz.trapmf(d_man, [70, 100, 100, 100])
    con["actual"] = fuzz.trimf(dout, [0.0, 0.0, 0.5])
    con["siguiente"] = fuzz.trimf(dout, [0.5, 1.0, 1.0])
    rules = [
        ctrl.Rule(ant["low"], con["actual"]),
        ctrl.Rule(ant["medium"] | ant["high"], con["siguiente"]),
    ]
    return ctrl.ControlSystemSimulation(ctrl.ControlSystem(rules))


def update_membership_functions(pred):
    """Ajusta los conjuntos con la predicción acotada a [0, 100]. Conserva las fórmulas originales."""
    p = float(np.clip(pred, 0, 100))
    low_ub = p * 0.5
    med_lb = p * 0.25
    med_ub = p * 0.75
    high_lb = low_ub
    ant = ctrl.Antecedent(d_man, "densidad")
    con = ctrl.Consequent(dout, "estado")
    ant["low"] = fuzz.trapmf(d_man, [0, 0, low_ub, p])
    ant["medium"] = fuzz.trapmf(d_man, [med_lb, med_ub, p, 100])
    ant["high"] = fuzz.trapmf(d_man, [high_lb, 100, 100, 100])
    con["actual"] = fuzz.trimf(dout, [0.0, 0.0, 0.5])
    con["siguiente"] = fuzz.trimf(dout, [0.5, 1.0, 1.0])
    rules = [
        ctrl.Rule(ant["low"], con["actual"]),
        ctrl.Rule(ant["medium"] | ant["high"], con["siguiente"]),
    ]
    sim = ctrl.ControlSystemSimulation(ctrl.ControlSystem(rules))
    return (sim, med_lb, med_ub)


def make_model():
    """MLP independiente por semáforo: ventana de 50, capa ReLU de 32 y salida lineal."""
    m = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(WINDOW_SIZE,)),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(1),
        ]
    )
    m.compile(optimizer="adam", loss="mse", run_eagerly=True)
    return m


def create_windowed_data(data, window_size):
    """Forma pares ventana/valor siguiente sin reordenar la serie temporal."""
    X, y = ([], [])
    for i in range(len(data) - window_size):
        X.append(data[i : i + window_size])
        y.append(data[i + window_size])
    return (np.array(X), np.array(y))


def run_neurofuzzy(max_steps=None):
    """Genera observaciones paso a paso. Una predicción ausente se escribe vacía hasta el paso 300."""
    tls = traci.trafficlight.getIDList()
    lanes_map = {
        tid: list(dict.fromkeys(traci.trafficlight.getControlledLanes(tid)))
        for tid in tls
    }
    buffers = {tid: deque(maxlen=BUFFER_SIZE) for tid in tls}
    models = {tid: make_model() for tid in tls}
    trained = {tid: False for tid in tls}
    sim_ctrls = {tid: defuzzify_static() for tid in tls}
    last_pred = {tid: None for tid in tls}
    med_lbs = {tid: initial_med_lb for tid in tls}
    med_ubs = {tid: initial_med_ub for tid in tls}
    step = 0
    while traci.simulation.getMinExpectedNumber() > 0 and (
        max_steps is None or step < max_steps
    ):
        traci.simulationStep()
        step += 1
        if step % 100 == 0:
            sys.stdout.write(f"\r[NeuroFuzzy] Step {step}")
            sys.stdout.flush()
        for tid, lanes in lanes_map.items():
            buf = buffers[tid]
            model = models[tid]
            dens = [get_density(l) for l in lanes]
            if dens:
                buf.append(np.mean(dens))
            if not trained[tid] and len(buf) > 2 * WINDOW_SIZE:
                X, y = create_windowed_data(list(buf), WINDOW_SIZE)
                if X.shape[0] > 0:
                    model.fit(
                        X,
                        y,
                        epochs=TRAIN_EPOCHS,
                        batch_size=TRAIN_BATCH,
                        callbacks=[tf.keras.callbacks.EarlyStopping(patience=1)],
                        verbose=0,
                    )
                    trained[tid] = True
            if trained[tid] and step % PREDICT_INTERVAL == 0:
                wnd = np.array(list(buf)[-WINDOW_SIZE:]).reshape(1, WINDOW_SIZE)
                pred = float(model.predict(wnd, verbose=0)[0][0])
                last_pred[tid] = pred
                sim_ctrls[tid], med_lbs[tid], med_ubs[tid] = (
                    update_membership_functions(pred)
                )
            ctrl = sim_ctrls[tid]
            cur_ph = traci.trafficlight.getPhase(tid)
            n_ph = len(traci.trafficlight.getAllProgramLogics(tid)[0].getPhases())
            for lane in lanes:
                d = get_density(lane)
                ctrl.input["densidad"] = d
                ctrl.compute()
                pv = last_pred[tid]
                yield (step, lane, d, pv, med_lbs[tid], med_ubs[tid])
                if ctrl.output["estado"] >= 0.5:
                    traci.trafficlight.setPhase(tid, (cur_ph + 1) % n_ph)
    print()


def run_fuzzy(max_steps=None):
    """Genera observaciones del controlador estático. y_pred=y_true es identidad, no un pronóstico."""
    tls = traci.trafficlight.getIDList()
    sim_ctrl = defuzzify_static()
    step = 0
    while traci.simulation.getMinExpectedNumber() > 0 and (
        max_steps is None or step < max_steps
    ):
        traci.simulationStep()
        step += 1
        if step % 100 == 0:
            sys.stdout.write(f"\r[Fuzzy] Step {step}")
            sys.stdout.flush()
        for tid in tls:
            cur_ph = traci.trafficlight.getPhase(tid)
            n_ph = len(traci.trafficlight.getAllProgramLogics(tid)[0].getPhases())
            for lane in traci.trafficlight.getControlledLanes(tid):
                d = get_density(lane)
                sim_ctrl.input["densidad"] = d
                sim_ctrl.compute()
                yield (step, lane, d, d, initial_med_lb, initial_med_ub)
                if sim_ctrl.output["estado"] >= 0.5:
                    traci.trafficlight.setPhase(tid, (cur_ph + 1) % n_ph)
    print()
