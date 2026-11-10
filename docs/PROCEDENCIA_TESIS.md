# Procedencia del código de la tesis

La LSTM de 50 unidades que describe la tesis se encuentra en `research/sumoneurofun/main.py` de la rama pública `main`. La MLP de 32 unidades corresponde al ejecutor que ya se había archivado en la versión 1.0.1. Son dos implementaciones de NeuroFuzzy con entradas distintas. [Definición de la LSTM importada](https://github.com/tono88/neurofuzzyTraffic/blob/17e8069ffcf8ab13fcf6b67957764f34445cc46e/research/sumoneurofun/main.py#L417-L425) · [Definición de la MLP en la versión 1.0.1](https://github.com/tono88/neurofuzzyTraffic/blob/0a4a4f66e46749a48dca086132efc54f819e5620/src/controllers.py#L79-L89).

El autor identifica `main.py` de la rama `main` de `tono88/SUMONeuroFun` como el código usado en la investigación. Se importaron los bytes del commit `57bf0d2629eca652f0238d751742a9a61d35b634`, concretamente [`main.py`](../research/sumoneurofun/main.py), [`run.sumocfg`](../research/sumoneurofun/run.sumocfg) y los siete archivos `joined*.xml` que referencia la configuración. La copia conserva las rutas relativas y la interfaz Tkinter. El archivo `lanes_con_semaforos.csv` es una salida del programa. [Commit de origen en el repositorio privado](https://github.com/tono88/SUMONeuroFun/commit/57bf0d2629eca652f0238d751742a9a61d35b634).

| Aspecto | Código de `SUMONeuroFun` importado | Ejecutor del paquete archivado 1.0.1 |
|---|---|---|
| Arquitectura | `LSTM(50, activation='relu')` seguida de `Dense(1)`. Las listas de rojo y verde referencian la misma instancia de red para todas las intersecciones. | MLP independiente por semáforo, con `Dense(32, activation='relu')` y `Dense(1)` lineal. |
| Entrada | Longitud de cola agrupada según el estado rojo o verde. La forma de entrada es un paso temporal por una característica. | Ventana de 50 observaciones de densidad media por semáforo, medida en vehículos por 100 m. |
| Actualización | El control difuso se evalúa cada 300 pasos. El bloque de entrenamiento y predicción se alcanza cada 3.000 pasos. Usa cinco épocas y lotes de 20. | Entrenamiento en línea con dos épocas y lotes de una observación. El código intenta predecir cada 300 pasos. |
| Semillas | El programa solicita tres corridas y genera una semilla pseudoaleatoria por corrida con `random.randint(1, 10000)`. La pasa a SUMO con `--seed`. No se dispone de una lista de las semillas de la investigación. | El ejecutor utiliza semillas de su propia configuración de referencia para comparar los modos Estandar, Fuzzy y NeuroFuzzy. |
| Fin de corrida | El bucle avanza mientras SUMO informa vehículos esperados. La configuración carece de un tiempo final explícito. | El mismo criterio por defecto. El ejecutor admite un límite opcional en pasos mediante `--max-steps`. |

El punto de entrada del paquete archivado es [`run_experiment.py`](../run_experiment.py). Importa `src/controllers.py`, cuya MLP deriva de [`original/main.py`](../original/main.py). La definición de la LSTM, su configuración SUMO y sus insumos se consultan en [`research/sumoneurofun/`](../research/sumoneurofun/). El DOI de la versión 1.0.1 identifica el ZIP con el ejecutor MLP y la ficha enlaza al repositorio público actualizado. [Registro de la versión 1.0.1](https://doi.org/10.5281/zenodo.23907073).

## Versión de SUMO y horizonte

- **Protocolo declarado en la tesis.** La sección 4.4.3 indica SUMO v1.15 con TraCI y 1.200 segundos por escenario. La denominación oficial de esa versión es **1.15.0**. La lectura «0.15» carece de respaldo en las fuentes inspeccionadas. [Historial oficial de SUMO 1.15.0](https://sumo.dlr.de/docs/ChangeLog/Changes_in_2022_releases.html#version_1150_08112022).
- **Registro en la rama fuente.** `sumo_log.txt` del commit de origen identifica una corrida con SUMO **1.16.0** que terminó en el tiempo simulado **5.159,00 s**. El registro por sí solo no vincula esa corrida con las tablas de la tesis. [Registro de ejecución de origen](https://github.com/tono88/SUMONeuroFun/blob/57bf0d2629eca652f0238d751742a9a61d35b634/sumo_log.txt).
- **Paquete publicado.** La [validación archivada](../validation/REPORT.md) corresponde a SUMO **1.16.0**. La configuración tampoco establece un fin a los **1.200 s**.

En el código importado, el primer bloque de entrenamiento y predicción de la LSTM está programado para el paso 3.000. Si el paso dura un segundo y se aplica el corte de 1.200 s declarado en la tesis, la corrida termina antes de alcanzar ese bloque. [Condición de entrenamiento en el código importado](../research/sumoneurofun/main.py#L486-L499).

La revisión se limitó a inspeccionar archivos y verificar la copia. La identificación del código proviene del autor. Queda por corroborar con registros de las corridas qué versión de SUMO, horizonte y configuración produjeron cada resultado de la tesis. Esta revisión no aporta resultados de simulación nuevos.
