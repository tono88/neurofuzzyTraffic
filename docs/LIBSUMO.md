# Ejecución CPU con libsumo y consultas optimizadas

Esta rama conserva el código de control usado en el lote iniciado el 3 de noviembre de 2026: longitudes de carril almacenadas durante la corrida, densidades reutilizadas solo dentro de cada paso y predicción directa mediante `model(wnd, training=False).numpy()`.

La caché de densidad se invalida inmediatamente después de cada `simulationStep`. La geometría de esta red es fija. Se mantienen semillas, orden de carriles y controles, fórmulas, precisión numérica, entrenamiento, dos épocas, lote de una observación y predicción cada 300 pasos.

## Instalación y ejecución

Ruta validada: Linux x86-64 / Ubuntu 24.04 en WSL2, Python 3.11, TensorFlow 2.15.1, SUMO y libsumo 1.16.0. Los paquetes libsumo 1.16.0 usados aquí no incluyen una rueda Windows para Python 3.11; no se debe actualizar silenciosamente el simulador para instalar otro paquete.

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements-libsumo.txt
export SUMO_HOME="$(.venv/bin/python -c 'import pathlib,sumo; print(pathlib.Path(sumo.__file__).resolve().parent)')"
export EXPERIMENT_BACKEND=libsumo
export EXPERIMENT_DEVICE=-1
.venv/bin/python run_experiment.py --plan
.venv/bin/python run_experiment.py --output results_libsumo
```

El comando completo aplica las 100 semillas publicadas a los tres controles. Para una sola combinación:

```bash
.venv/bin/python run_experiment.py --seed 179 --modes NeuroFuzzy --output results_libsumo_179
```

`EXPERIMENT_BACKEND=traci` conserva la comunicación convencional. Cuando no se especifica backend, el valor predeterminado es TraCI. El backend y la versión de libsumo forman parte del entorno registrado y de la firma que verifica `--resume`. Cada combinación usa un proceso independiente. La integración sustituye el módulo importado por los controladores por libsumo, sin cambiar las funciones científicas entre backends.

`EXPERIMENT_DEVICE=-1` selecciona CPU y es el valor predeterminado. La selección explícita `0` verifica que exista GPU y permite crecimiento de memoria; requiere un entorno GPU compatible y no se usa en el lote CPU. No garantiza igualdad entre CPU y GPU.

## Verificación realizada el 3 de noviembre de 2026

Las pruebas cortas compararon TraCI y libsumo en la misma plataforma Linux y CPU:

| Control | Semilla | Pasos | TraCI | libsumo | Salidas iguales |
|---|---:|---:|---:|---:|---|
| Estándar | 179 | 100 | 3,01 s | 1,36 s | Sí |
| Fuzzy | 179 | 305 | 31,07 s | 15,54 s | Sí |
| NeuroFuzzy | 179 | 1.000 | 73,98 s | 46,41 s | Sí |
| NeuroFuzzy | 323 | 1.000 | 63,07 s | 38,54 s | Sí |

Una réplica completa NeuroFuzzy con semilla 179 terminó en 123.334 pasos y cero vehículos pendientes. Se comparó con la réplica original Windows/TraCI/CPU archivada: coincidieron exactamente los datos CSV (SHA-256), los valores de ambas métricas JSON y la estructura, atributos y datos de `queue.xml`, `tripinfo.xml` y `e1_output.xml`, excluyendo comentarios de ejecución. También coincidió el número de pasos.

La réplica original tardó aproximadamente 3 h 46 min. El tiempo medido del proceso libsumo fue de unos 30 min; el intervalo entre fechas del manifiesto es de unos 31 min. Son una observación por variante, no una garantía para otras semillas o equipos. La equivalencia completa demostrada cubre la semilla 179; las otras pruebas son recortadas. Las mediciones de GPU anteriores detectaron diferencias pequeñas respecto a CPU.

Las salidas de simulación se generan localmente y no se añaden a esta rama. El panel local, las rutas específicas de D: y el ejecutor de compresión del equipo no son necesarios para ejecutar este código y no forman parte de esta modificación del controlador.

## Comprobaciones del paquete

```bash
.venv/bin/python scripts/verify_package.py
.venv/bin/python -m unittest discover -s tests -v
```

Referencias: [libsumo](https://sumo.dlr.de/docs/Libsumo.html), [Keras: predicción directa para entradas pequeñas](https://keras.io/2/api/models/model_training_apis/).
