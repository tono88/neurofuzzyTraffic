# Guía del ejecutor MLP del paquete 1.0.1

Esta guía documenta el ejecutor de la versión 1.0.1 archivada en Zenodo. Su modo NeuroFuzzy usa una MLP de 32 unidades ReLU y 50 observaciones de densidad media como entrada. Las instrucciones y la validación de esta guía se refieren a ese paquete. El código LSTM de la tesis se identifica en la [nota de procedencia](PROCEDENCIA_TESIS.md). [Archivo 1.0.1](https://doi.org/10.5281/zenodo.23907073).

## 1. Qué contiene

| Ruta | Contenido y propósito |
|---|---|
| `original/` | Implementaciones de control, utilidades, modelos, resultados y figuras del proyecto. |
| `original/MANIFEST.json` | Inventario de archivos con tamaño y huella SHA-256. |
| `scenarios/bologna/` | Red, demanda privada, autobuses, paradas, detectores, tipos vehiculares, planes semafóricos y configuración originales. |
| `src/controllers.py` | Funciones de control derivadas de `original/main.py`, con documentación y salida incremental. |
| `run_experiment.py` | Ejecutor del paquete MLP con diseño pareado, procesos independientes y registro del entorno. |
| `scripts/verify_package.py` | Verificación de originales, escenario y concordancia TXT/CSV. |
| `scripts/summarize_results.py` | Resumen de movilidad: una fila por modo y semilla. |
| `tests/` | Casos de validación para predicciones ausentes, métricas, duplicados y aislamiento de archivos. |
| `validation/` | Resultados de las comprobaciones de integridad, ejecución y repetibilidad. |
| `docs/` | Metodología, inventario de archivos y publicación del software. |
| `CITATION.cff`, `.zenodo.json` | Metadatos de citación, autoría, supervisión y archivo científico. |

## 2. Escenario distribuido

Solo se distribuye **un escenario geográfico**, Bologna `joined`; los tres modos son condiciones de control sobre ese mismo escenario.

| Archivo | Función |
|---|---|
| `run.sumocfg` | Vincula todos los insumos. |
| `joined_buslanes.net.xml` | Red vial y carriles de autobús: 766 elementos `edge` y 1049 carriles, incluidos los internos; 13 programas `tlLogic` en este archivo. |
| `joined.rou.xml` | 11 000 vehículos, salidas programadas entre 0 y 3598 s. |
| `joined_busses.rou.xml` | 176 autobuses, salidas programadas entre 0 y 3600 s. |
| `joined_bus_stops.add.xml` | Paradas de autobús. |
| `joined_detectors.add.xml` | Detectores de lazo y sus intervalos. |
| `joined_vtypes.add.xml` | Tipos de vehículo. |
| `joined_tls.add.xml` | Planes semafóricos adicionales. |

La demanda ya está definida en los XML: **no se generan nuevas rutas** ni se llama a `randomTrips.py`. La semilla modifica la aleatoriedad interna de SUMO y, en NeuroFuzzy, la de la red neuronal. La configuración no fija una hora final: una corrida completa continúa hasta que no quedan vehículos esperados. La demanda de una hora no implica que la simulación termine exactamente a los 3600 s.

Origen de referencia: [escenarios oficiales SUMO — Bologna](https://sumo.dlr.de/docs/Data/Scenarios.html#bologna) y [DLR-TS/sumo-scenarios](https://github.com/DLR-TS/sumo-scenarios/tree/main/bologna). La red, la demanda y los planes semafóricos se distribuyen en `scenarios/bologna/`.

## 3. Requisitos e instalación

- Python **3.11** de 64 bits.
- Eclipse SUMO **1.16.0** instalado por separado. Versión del simulador utilizada en las pruebas de ejecución.
- Dependencias de referencia fijadas en `requirements.txt`. Este archivo define el entorno Python de ejecución.
- CPU para la ruta de referencia determinista. Cada corrida usa un proceso Python nuevo y un hilo de ejecución TensorFlow.

Desde la carpeta del repositorio, en **Windows PowerShell**:

```powershell
python -m venv .venv
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
$env:SUMO_HOME = 'C:\Program Files (x86)\Eclipse\Sumo'
& .\.venv\Scripts\python.exe scripts/verify_package.py
```

En **Linux**, con SUMO 1.16.0 instalado y su carpeta `tools` disponible:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
export SUMO_HOME=/ruta/a/sumo
.venv/bin/python scripts/verify_package.py
```

`SUMO_HOME` debe contener `bin/sumo` (o `bin/sumo.exe`) y `tools/traci`. También puede indicarse con `--sumo-home`. TraCI y sumolib se toman de esa instalación para mantener su correspondencia con el simulador. No hace falta instalar otro TraCI con pip.

Para ejecutar solo `Estandar`, planificar o verificar archivos basta la biblioteca estándar de Python y la instalación de SUMO. Los dos modos difusos usan el entorno científico completo.

## 4. Uso del ejecutor MLP

En los ejemplos siguientes, `python` representa el Python del entorno creado; en Windows puede sustituirse por `& .\.venv\Scripts\python.exe`.

### Inspeccionar el diseño sin simular

```bash
python run_experiment.py --plan
```

Muestra las corridas previstas por el ejecutor MLP para su configuración de referencia, sin iniciar SUMO. Para seleccionar los dos modos presentes en el bloque principal de `original/main.py`, use:

```bash
python run_experiment.py --modes Fuzzy NeuroFuzzy --plan
```

La línea base Estandar se incorpora al paquete MLP para comparar movilidad.

### Prueba corta antes del experimento

```bash
python run_experiment.py --seed 179 --modes Estandar --max-steps 10 --output results_smoke_standard
python run_experiment.py --seed 179 --modes Fuzzy NeuroFuzzy --max-steps 305 --output results_smoke_control
```

Una prueba de 305 pasos cruza el entrenamiento inicial de NeuroFuzzy y su primera predicción, prevista en el paso 300. `--max-steps` recorta la corrida y el manifiesto la marca `truncated` si aún quedan vehículos. **Estas pruebas no son resultados completos del experimento.**

### Ejecutar el diseño de referencia del paquete MLP

```bash
python run_experiment.py --output results
```

El ejecutor aplica, en orden, cada semilla a Estandar, Fuzzy y NeuroFuzzy. El orden de modos no contamina el estado neuronal porque cada combinación corre en un proceso independiente. La ejecución es secuencial; el tiempo y espacio totales dependen del tráfico y del equipo. Los CSV de detalle pueden crecer considerablemente.

Para una corrida completa por modo con la primera semilla:

```bash
python run_experiment.py --limit 1 --output results_pilot
```

Para omitir corridas ya completadas y verificadas:

```bash
python run_experiment.py --output results --resume
```

La reanudación comprueba estado `completed`, versiones, parámetros, huellas de entrada y huellas de salida. Rechaza destinos parciales, recortados, modificados o de otra configuración. Para repetirlos, use una carpeta de salida nueva; no se sobrescriben archivos existentes.

### Resumir resultados

```bash
python scripts/summarize_results.py results
```

Produce `results/summary.csv` con una fila por corrida completa: modo, semilla, pasos, tiempo simulado, vehículos pendientes, viajes llegados y promedios de duración, espera, pérdida de tiempo y longitud de ruta. La unidad experimental es la corrida, no cada fila de carril. No mezcle pruebas cortas con el análisis principal.

## 5. Salidas de cada corrida

Por motivos de almacenamiento, no se adjuntan los archivos completos de salida de las simulaciones. Estos se generan localmente al ejecutar los comandos indicados.

```text
results/
  plan.json
  Estandar/seed_179/
  Fuzzy/seed_179/
  NeuroFuzzy/seed_179/
    job.json
    manifest.json
    console.log
    run.sumocfg
    detectors.add.xml
    sumo.log
    tripinfo.xml
    queue.xml
    e1_output.xml
    details.csv
    prediction_metrics.json
    traffic_metrics.json
```

`details.csv` y `prediction_metrics.json` solo existen en los modos Fuzzy y NeuroFuzzy.

| Campo de detalle | Interpretación |
|---|---|
| `step` | Número de paso, con primer paso igual a 1. |
| `lane_id` | Identificador SUMO del carril. |
| `y_true` | Número de vehículos por 100 m de carril: `100 × n / longitud`. No es porcentaje de ocupación. |
| `y_pred` | NeuroFuzzy: última predicción de densidad media de los carriles del semáforo, mantenida entre actualizaciones; vacía antes de la primera predicción. Fuzzy: el mismo valor observado, según el registro de observaciones del controlador estático. |
| `med_lb`, `med_ub` | Umbrales utilizados para clasificar Bajo/Medio/Alto en esa observación. |

`manifest.json` registra la semilla, modo, parámetros, versiones, huellas SHA-256, comando SUMO, estado, pasos, vehículos pendientes y fechas UTC. Los estados `failed` y `truncated` impiden confundir una ejecución parcial con una completa.

Las métricas de Fuzzy basadas en `y_pred=y_true` describen una identidad y **no demuestran precisión predictiva**. NeuroFuzzy contrasta una predicción media del semáforo con densidades actuales de carril; no es por sí solo una evaluación prospectiva a un horizonte común. Consulte [METODOLOGIA.md](METODOLOGIA.md) antes de interpretar RMSE, MAE, MAPE, R² o matrices de confusión.

## 6. Verificación

```bash
python scripts/verify_package.py
python -m unittest discover -s tests -v
```

La automatización de GitHub verifica la integridad del paquete, las pruebas del ejecutor y la planificación del diseño de referencia. Los resultados de las pruebas de ejecución se presentan en [validation/REPORT.md](../validation/REPORT.md).
