#!/usr/bin/env python
"""
Multi-mode traffic control para SUMO: NeuroFuzzy, Fuzzy y Estándar con:
 - Una MLP por intersección (semáforo).
 - Guardado de detalles (step, lane_id, y_true, y_pred, med_lb, med_ub) en CSV por chunks.
 - Clasificación y matrices de confusión OFFLINE tras la simulación.
 - Métricas de regresión en Excel (filtrando NaNs para evitar errores).
 - Predicción sólo cada PREDICT_INTERVAL y clasificación por umbral.
"""
import os
import sys
import optparse
import logging
import csv
from collections import deque

# Suprimir logs de TF y tf.data
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.data.experimental.enable_debug_mode()
tf.get_logger().setLevel(logging.ERROR)

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    confusion_matrix
)
import pandas as pd

# --- Logging ---
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
tf.config.run_functions_eagerly(True)

# --- Init SUMO/TraCI ---
def init_sumo():
    if 'SUMO_HOME' not in os.environ:
        sys.exit("ERROR: Declare SUMO_HOME apuntando a SUMO.")
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
    from sumolib import checkBinary
    import traci
    return checkBinary, traci

checkBinary, traci = init_sumo()

# --- CLI options ---
def get_options():
    p = optparse.OptionParser()
    p.add_option("--nogui", action="store_true", default=True,
                 help="Ejecutar SUMO sin GUI")
    p.add_option("--runs", dest="runs", type="int", default=1,
                 help="Número de réplicas de simulación")
    opts, _ = p.parse_args()
    return opts

# --- Dominio fuzzy ---
d_man = np.arange(0, 101, 1)
dout  = np.arange(0, 1.01, 0.01)

# Umbrales iniciales
initial_med_lb = 30.0
initial_med_ub = 70.0

# --- Obtener densidad (0-100) ---
def get_density(lane_id):
    length = traci.lane.getLength(lane_id)
    return (len(traci.lane.getLastStepVehicleIDs(lane_id)) / length * 100.0
            if length > 0 else 0.0)

# --- Fuzzy estático ---
def defuzzify_static():
    ant = ctrl.Antecedent(d_man, 'densidad')
    con = ctrl.Consequent(dout, 'estado')
    ant['low']    = fuzz.trapmf(d_man, [0, 0, 30, 50])
    ant['medium'] = fuzz.trapmf(d_man, [30, 50, 70, 100])
    ant['high']   = fuzz.trapmf(d_man, [70,100,100,100])
    con['actual']    = fuzz.trimf(dout, [0.0,0.0,0.5])
    con['siguiente'] = fuzz.trimf(dout, [0.5,1.0,1.0])
    rules = [
        ctrl.Rule(ant['low'],    con['actual']),
        ctrl.Rule(ant['medium']|ant['high'], con['siguiente'])
    ]
    return ctrl.ControlSystemSimulation(ctrl.ControlSystem(rules))

# --- Ajuste dinámico según predicción ---
def update_membership_functions(pred):
    p = float(np.clip(pred, 0, 100))
    low_ub  = p * 0.5
    med_lb  = p * 0.25
    med_ub  = p * 0.75
    high_lb = low_ub
    ant = ctrl.Antecedent(d_man, 'densidad')
    con = ctrl.Consequent(dout, 'estado')
    ant['low']    = fuzz.trapmf(d_man, [0,0,low_ub,p])
    ant['medium'] = fuzz.trapmf(d_man, [med_lb,med_ub,p,100])
    ant['high']   = fuzz.trapmf(d_man, [high_lb,100,100,100])
    con['actual']    = fuzz.trimf(dout, [0.0,0.0,0.5])
    con['siguiente'] = fuzz.trimf(dout, [0.5,1.0,1.0])
    rules = [
        ctrl.Rule(ant['low'],    con['actual']),
        ctrl.Rule(ant['medium']|ant['high'], con['siguiente'])
    ]
    sim = ctrl.ControlSystemSimulation(ctrl.ControlSystem(rules))
    return sim, med_lb, med_ub

# --- Parámetros NeuroFuzzy ---
WINDOW_SIZE      = 50
BUFFER_SIZE      = 200
PREDICT_INTERVAL = 300
TRAIN_EPOCHS     = 2
TRAIN_BATCH      = 1

# --- Factory de MLP ---
def make_model():
    m = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(WINDOW_SIZE,)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    m.compile(optimizer='adam', loss='mse', run_eagerly=True)
    return m

# --- Utils de datos y métricas (filtrando NaNs) ---
def create_windowed_data(data, window_size):
    X,y = [],[]
    for i in range(len(data)-window_size):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size])
    return np.array(X), np.array(y)

def compute_regression_metrics(y_true, y_pred):
    yt = np.array(y_true, dtype=float)
    yp = np.array(y_pred, dtype=float)
    mask = np.isfinite(yt) & np.isfinite(yp)
    yt, yp = yt[mask], yp[mask]
    if len(yt)==0:
        return {'RMSE': np.nan, 'MAE': np.nan, 'MAPE (%)': np.nan, 'R2': np.nan}
    rmse = np.sqrt(mean_squared_error(yt, yp))
    mae  = mean_absolute_error(yt, yp)
    mape = np.mean(np.abs((yt - yp) / np.where(yt!=0, yt, 1))) * 100
    r2   = r2_score(yt, yp)
    return {'RMSE': rmse, 'MAE': mae, 'MAPE (%)': mape, 'R2': r2}

def plot_confusion_matrix(cm, classes, filename, title):
    fig, ax = plt.subplots(figsize=(5,5))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(len(classes)),
           yticks=np.arange(len(classes)),
           xticklabels=classes, yticklabels=classes,
           xlabel='Predicción', ylabel='Valor real', title=title)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i,j]), ha='center',
                    color='white' if cm[i,j]>cm.max()/2 else 'black')
    plt.tight_layout()
    fig.savefig(filename, dpi=300)
    plt.close(fig)

# --- Ejecuta un modo y guarda detalles por chunks ---
def run_mode(mode_name, mode_func, seed, out_trip, out_queue,
             aggregated, run_metrics):
    traci.start([
        checkBinary('sumo'), '-c', 'run.sumocfg',
        '--seed', str(seed),
        '--tripinfo-output', out_trip,
        '--queue-output',   out_queue
    ])
    # Simulación
    y_true, y_pred, steps, lane_ids, med_lb_list, med_ub_list = mode_func()

    # Escritura en CSV por chunks
    os.makedirs(mode_name, exist_ok=True)
    csv_path = os.path.join(mode_name, f'details_seed{seed}.csv')
    # Cabecera
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['step','lane_id','y_true','y_pred','med_lb','ub_real'])

    chunksize = 10_000
    total = len(steps)
    for start in range(0, total, chunksize):
        end = min(start + chunksize, total)
        with open(csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            for i in range(start, end):
                writer.writerow([
                    steps[i],
                    lane_ids[i],
                    y_true[i],
                    y_pred[i],
                    med_lb_list[i],
                    med_ub_list[i]
                ])

    # Métricas agregadas
    m = compute_regression_metrics(y_true, y_pred)
    run_metrics[mode_name].append({'seed': seed, **m})
    aggregated[mode_name]['y_true'].extend(y_true)
    aggregated[mode_name]['y_pred'].extend(y_pred)
    traci.close()

# --- MODO NeuroFuzzy ---
def run_neurofuzzy():
    tls       = traci.trafficlight.getIDList()
    lanes_map = {tid: list(dict.fromkeys(traci.trafficlight.getControlledLanes(tid)))
                 for tid in tls}
    buffers = {tid: deque(maxlen=BUFFER_SIZE) for tid in tls}
    models  = {tid: make_model()         for tid in tls}
    trained = {tid: False               for tid in tls}
    sim_ctrls = {tid: defuzzify_static() for tid in tls}
    last_pred = {tid: None for tid in tls}
    med_lbs   = {tid: initial_med_lb for tid in tls}
    med_ubs   = {tid: initial_med_ub for tid in tls}

    y_true, y_pred = [], []
    steps, lane_ids = [], []
    med_lb_list, med_ub_list = [], []

    step = 0
    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        step += 1
        if step % 100 == 0:
            sys.stdout.write(f"\r[NeuroFuzzy] Step {step}")
            sys.stdout.flush()

        for tid, lanes in lanes_map.items():
            buf   = buffers[tid]
            model = models[tid]

            dens = [get_density(l) for l in lanes]
            if dens:
                buf.append(np.mean(dens))

            if not trained[tid] and len(buf) > 2 * WINDOW_SIZE:
                X, y = create_windowed_data(list(buf), WINDOW_SIZE)
                if X.shape[0] > 0:
                    model.fit(X, y,
                              epochs=TRAIN_EPOCHS,
                              batch_size=TRAIN_BATCH,
                              callbacks=[tf.keras.callbacks.EarlyStopping(patience=1)],
                              verbose=0)
                    trained[tid] = True

            if trained[tid] and step % PREDICT_INTERVAL == 0:
                wnd  = np.array(list(buf)[-WINDOW_SIZE:]).reshape(1, WINDOW_SIZE)
                pred = float(model.predict(wnd, verbose=0)[0][0])
                last_pred[tid] = pred
                sim_ctrls[tid], med_lbs[tid], med_ubs[tid] = update_membership_functions(pred)

            ctrl = sim_ctrls[tid]
            cur_ph = traci.trafficlight.getPhase(tid)
            n_ph   = len(traci.trafficlight.getAllProgramLogics(tid)[0].getPhases())
            for lane in lanes:
                d = get_density(lane)
                ctrl.input['densidad'] = d
                ctrl.compute()
                pv = last_pred[tid] if trained[tid] else d

                y_true.append(d)
                y_pred.append(pv)
                steps.append(step)
                lane_ids.append(lane)
                med_lb_list.append(med_lbs[tid])
                med_ub_list.append(med_ubs[tid])

                if ctrl.output['estado'] >= 0.5:
                    traci.trafficlight.setPhase(tid, (cur_ph + 1) % n_ph)

    print()
    return y_true, y_pred, steps, lane_ids, med_lb_list, med_ub_list

# --- MODO Fuzzy ---
def run_fuzzy():
    tls      = traci.trafficlight.getIDList()
    sim_ctrl = defuzzify_static()

    y_true, y_pred = [], []
    steps, lane_ids = [], []
    med_lb_list, med_ub_list = [], []

    step = 0
    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        step += 1
        if step % 100 == 0:
            sys.stdout.write(f"\r[Fuzzy] Step {step}")
            sys.stdout.flush()

        for tid in tls:
            cur_ph = traci.trafficlight.getPhase(tid)
            n_ph   = len(traci.trafficlight.getAllProgramLogics(tid)[0].getPhases())
            for lane in traci.trafficlight.getControlledLanes(tid):
                d = get_density(lane)
                sim_ctrl.input['densidad'] = d
                sim_ctrl.compute()

                y_true.append(d)
                y_pred.append(d)
                steps.append(step)
                lane_ids.append(lane)
                med_lb_list.append(initial_med_lb)
                med_ub_list.append(initial_med_ub)

                if sim_ctrl.output['estado'] >= 0.5:
                    traci.trafficlight.setPhase(tid, (cur_ph + 1) % n_ph)

    print()
    return y_true, y_pred, steps, lane_ids, med_lb_list, med_ub_list

# --- Main ---
if __name__ == '__main__':
    opts = get_options()
    modes = [('Fuzzy', run_fuzzy), ('NeuroFuzzy', run_neurofuzzy)]

    aggregated  = {m: {'y_true': [], 'y_pred': []} for m,_ in modes}
    run_metrics = {m: [] for m,_ in modes}

    for m,_ in modes:
        os.makedirs(m, exist_ok=True)
    seeds = np.random.randint(1,10000, size=opts.runs)
    logger.info("Semillas generadas: %s", seeds.tolist())

    for mode_name, mode_func in modes:
        for seed in seeds:
            out_trip  = os.path.join(mode_name, f'tripinfo_{seed}.xml')
            out_queue = os.path.join(mode_name, f'queue_{seed}.xml')
            run_mode(mode_name, mode_func, seed,
                     out_trip, out_queue,
                     aggregated, run_metrics)

    # Escribir métricas en Excel
    with pd.ExcelWriter('resultados_metricas.xlsx', engine='xlsxwriter') as writer:
        for m,_ in modes:
            agg = compute_regression_metrics(
                aggregated[m]['y_true'],
                aggregated[m]['y_pred']
            )
            pd.DataFrame(agg, index=[0]).T.to_excel(writer, sheet_name=f'Agg_{m}')
            pd.DataFrame(run_metrics[m]).set_index('seed').T.to_excel(writer, sheet_name=f'Run_{m}')

    # OFFLINE: matrices de confusión por lane y global
    classes = ['Bajo','Medio','Alto']
    for mode_name,_ in modes:
        df_list = []
        for f in os.listdir(mode_name):
            if f.startswith('details_seed') and f.endswith('.csv'):
                df_list.append(pd.read_csv(os.path.join(mode_name, f)))
        if not df_list:
            logger.warning("No se encontraron archivos de detalle en %s", mode_name)
            continue

        df = pd.concat(df_list, ignore_index=True)
        df['true_cls'] = df.apply(
            lambda r: 'Bajo'  if r['y_true'] < r['med_lb']
                      else 'Medio' if r['y_true'] < r['med_ub']
                      else 'Alto',
            axis=1
        )
        df['pred_cls'] = df.apply(
            lambda r: 'Bajo'  if r['y_pred'] < r['med_lb']
                      else 'Medio' if r['y_pred'] < r['med_ub']
                      else 'Alto',
            axis=1
        )

        # Por cada intersección (lane)
        for lane in df['lane_id'].unique():
            sub = df[df['lane_id']==lane]
            cm = confusion_matrix(sub['true_cls'], sub['pred_cls'], labels=classes)
            plot_confusion_matrix(
                cm, classes,
                os.path.join(mode_name, f'cm_{mode_name}_lane_{lane}.png'),
                f'CM {mode_name} Lane={lane}'
            )
        # Global
        cmg = confusion_matrix(df['true_cls'], df['pred_cls'], labels=classes)
        plot_confusion_matrix(
            cmg, classes,
            os.path.join(mode_name, f'cm_{mode_name}_global.png'),
            f'CM {mode_name} Global'
        )

    logger.info("Matrices de confusión offline generadas por lane y global.")
