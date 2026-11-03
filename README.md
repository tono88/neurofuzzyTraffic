# Bologna SUMO: código, escenario y semillas para reproducción

Paquete de investigación para ejecutar control semafórico **Fuzzy**, **NeuroFuzzy** y una línea base **Estandar** en el escenario Bologna `joined` de SUMO. Incluye el código fuente, los archivos del escenario, **100 semillas explícitas** y un ejecutor que registra cada corrida por separado.

**Autor:** Estuardo Antonio Sandoval Acevedo · [ORCID 0009-0005-2146-6699](https://orcid.org/0009-0005-2146-6699).  
**Dirección de tesis:** David Escuin · [ORCID 0000-0003-2062-8092](https://orcid.org/0000-0003-2062-8092).

**Repositorio:** [tono88/neurofuzzyTraffic](https://github.com/tono88/neurofuzzyTraffic). **DOI de la versión 1.0.1:** [10.5281/zenodo.23907073](https://doi.org/10.5281/zenodo.23907073).

## 1. Qué contiene

| Ruta | Contenido y propósito |
|---|---|
| `original/` | Implementaciones de control, utilidades, modelos, resultados y figuras del proyecto. |
| `original/MANIFEST.json` | Inventario de archivos con tamaño y huella SHA-256. |
| `scenarios/bologna/` | Red, demanda privada, autobuses, paradas, detectores, tipos vehiculares, planes semafóricos y configuración originales. |
| `src/controllers.py` | Funciones de control derivadas de `main.py`, con documentación y salida incremental. |
| `run_experiment.py` | Diseño pareado, semillas fijas, procesos independientes, control de salidas y registro del entorno. |
| `config/seeds.txt` | Fuente operativa de las 100 semillas, una por línea. |
| `config/seeds.csv` | Índice de réplica y semilla para inspección o importación. |
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

## 4. Reproducción paso a paso

En los ejemplos siguientes, `python` representa el Python del entorno creado; en Windows puede sustituirse por `& .\.venv\Scripts\python.exe`.

### Inspeccionar el diseño sin simular

```bash
python run_experiment.py --plan
```

Muestra **100 semillas × 3 modos = 300 corridas planificadas**. No genera semillas al azar ni inicia SUMO. Para reproducir únicamente los dos modos presentes en el bloque principal de `main.py`:

```bash
python run_experiment.py --modes Fuzzy NeuroFuzzy --plan
```

Esto planifica **200 corridas**. La línea base Estandar se incorpora para comparar movilidad y se identifica como una adición del paquete.

### Prueba corta antes del experimento

```bash
python run_experiment.py --seed 179 --modes Estandar --max-steps 10 --output results_smoke_standard
python run_experiment.py --seed 179 --modes Fuzzy NeuroFuzzy --max-steps 305 --output results_smoke_control
```

Una prueba de 305 pasos cruza el entrenamiento inicial de NeuroFuzzy y su primera predicción, prevista en el paso 300. `--max-steps` recorta la corrida y el manifiesto la marca `truncated` si aún quedan vehículos. **Estas pruebas no son resultados completos del experimento.**

### Ejecutar las 100 semillas

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

Las métricas de Fuzzy basadas en `y_pred=y_true` describen una identidad y **no demuestran precisión predictiva**. NeuroFuzzy contrasta una predicción media del semáforo con densidades actuales de carril; no es por sí solo una evaluación prospectiva a un horizonte común. Consulte [METODOLOGIA.md](docs/METODOLOGIA.md) antes de interpretar RMSE, MAE, MAPE, R² o matrices de confusión.

## 6. Las 100 semillas

El experimento utiliza las siguientes **100 semillas fijas**. La lista se distribuye en `config/seeds.txt` y `config/seeds.csv`, en el mismo orden de la tabla. Cada semilla se aplica a los tres modos de control para realizar la comparación pareada.

| Réplica | Semilla | Réplica | Semilla | Réplica | Semilla | Réplica | Semilla |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 179 | 26 | 3046 | 51 | 5448 | 76 | 7539 |
| 2 | 195 | 27 | 3410 | 52 | 5526 | 77 | 7604 |
| 3 | 323 | 28 | 3427 | 53 | 5546 | 78 | 7715 |
| 4 | 452 | 29 | 3438 | 54 | 5552 | 79 | 7820 |
| 5 | 543 | 30 | 3652 | 55 | 5773 | 80 | 8427 |
| 6 | 620 | 31 | 3776 | 56 | 5781 | 81 | 8501 |
| 7 | 775 | 32 | 3922 | 57 | 5783 | 82 | 8533 |
| 8 | 936 | 33 | 4017 | 58 | 5862 | 83 | 8739 |
| 9 | 1106 | 34 | 4102 | 59 | 5898 | 84 | 8869 |
| 10 | 1228 | 35 | 4122 | 60 | 5922 | 85 | 8878 |
| 11 | 1395 | 36 | 4295 | 61 | 5993 | 86 | 8973 |
| 12 | 1560 | 37 | 4308 | 62 | 6211 | 87 | 8988 |
| 13 | 1794 | 38 | 4318 | 63 | 6241 | 88 | 9017 |
| 14 | 1948 | 39 | 4443 | 64 | 6256 | 89 | 9035 |
| 15 | 2035 | 40 | 4512 | 65 | 6412 | 90 | 9086 |
| 16 | 2078 | 41 | 4519 | 66 | 6543 | 91 | 9126 |
| 17 | 2111 | 42 | 4585 | 67 | 6746 | 92 | 9453 |
| 18 | 2187 | 43 | 4623 | 68 | 6809 | 93 | 9457 |
| 19 | 2264 | 44 | 4700 | 69 | 6848 | 94 | 9458 |
| 20 | 2565 | 45 | 4783 | 70 | 7062 | 95 | 9490 |
| 21 | 2646 | 46 | 4784 | 71 | 7228 | 96 | 9552 |
| 22 | 2651 | 47 | 5000 | 72 | 7313 | 97 | 9568 |
| 23 | 2652 | 48 | 5245 | 73 | 7352 | 98 | 9667 |
| 24 | 2695 | 49 | 5247 | 74 | 7416 | 99 | 9917 |
| 25 | 2727 | 50 | 5348 | 75 | 7508 | 100 | 9977 |

## 7. Verificación

```bash
python scripts/verify_package.py
python -m unittest discover -s tests -v
```

La automatización de GitHub verifica la integridad del paquete, las pruebas del ejecutor y el diseño de 300 corridas. Los resultados de las pruebas de ejecución se presentan en [validation/REPORT.md](validation/REPORT.md).

## 8. Citar y obtener el DOI

La versión **1.0.1** está archivada y publicada en [Zenodo](https://zenodo.org/records/23907073). Su DOI de versión es **[10.5281/zenodo.23907073](https://doi.org/10.5281/zenodo.23907073)**. Los metadatos de citación están en `CITATION.cff` y `.zenodo.json`.

> Sandoval Acevedo, E. A. (2026). *Bologna SUMO: código, escenario y 100 semillas para reproducción de control semafórico* (versión 1.0.1) [Software]. Zenodo. https://doi.org/10.5281/zenodo.23907073

Para citar esta distribución se utiliza el DOI de versión. El DOI de concepto [10.5281/zenodo.23907072](https://doi.org/10.5281/zenodo.23907072) agrupa las versiones del software. Los enlaces del repositorio y del depósito están registrados en [PUBLICATION.json](PUBLICATION.json).

## 9. Licencias y procedencia

El código del proyecto y su documentación se distribuyen bajo [MIT](LICENSE). Los archivos de terceros mantienen sus condiciones: escenario SUMO bajo los términos de su distribución de origen y `original/randomTrips.py` con su aviso **EPL-2.0 OR GPL-2.0-or-later**. Se incluyen los textos oficiales disponibles en `licenses/` y el detalle en [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). La licencia MIT no cambia los derechos de los componentes de terceros ni los de las dependencias instaladas por separado.

## 10. Referencias del escenario y del entorno

- Bieker, L., Krajzewicz, D., Morra, A., Michelacci, C. y Cartolano, F. (2014). *Traffic simulation for all: a real world traffic scenario from the city of Bologna*. SUMO2014, pp. 19–26. [Registro DLR y escenario](https://sumo.dlr.de/docs/Data/Scenarios.html#bologna).
- [SUMO: control de aleatoriedad mediante `--seed`](https://sumo.dlr.de/docs/Simulation/Randomness.html).
- [TensorFlow: determinismo de operaciones](https://www.tensorflow.org/api_docs/python/tf/config/experimental/enable_op_determinism).
- [Zenodo: habilitar el repositorio](https://help.zenodo.org/docs/github/enable-repository/) y [archivar una release](https://help.zenodo.org/docs/github/archive-software/github-upload/).
