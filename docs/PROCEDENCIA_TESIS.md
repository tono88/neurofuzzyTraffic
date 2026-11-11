# Procedencia del código LSTM

El autor identifica el `main.py` de `tono88/SUMONeuroFun` como el código de la investigación. La importación se realizó desde el commit `57bf0d2629eca652f0238d751742a9a61d35b634`. El archivo y sus ocho insumos se copiaron en `research/sumoneurofun/`. [Origen](https://github.com/tono88/SUMONeuroFun/tree/57bf0d2629eca652f0238d751742a9a61d35b634) · [Importación pública original](https://github.com/tono88/neurofuzzyTraffic/tree/17e8069ffcf8ab13fcf6b67957764f34445cc46e/research/sumoneurofun).

## Confirmaciones del autor y revisión del código

El 11 de noviembre de 2026 el autor confirmó **SUMO 1.15.0** como versión de la investigación y aclaró que las **100 iteraciones** se refieren a **100 corridas de simulación**. También precisó que los **1.200 s** describían una corrida optimista sin el control NeuroFuzzy.

| Elemento | Evidencia y estado |
|---|---|
| Número de corridas | La copia importada usaba `range(3)` y una barra de progreso con máximo 3. Esta revisión introduce `NUM_SIMULATIONS = 100` para ambos usos. |
| SUMO | La versión 1.15.0 procede de la confirmación del autor. El código localiza el ejecutable en el entorno. Las versiones Python y dependencias de las corridas originales siguen pendientes de corroborar. |
| Horizonte de la LSTM | La configuración distribuida termina conforme al criterio de vehículos esperados. La aclaración sobre 1.200 s se refiere a la corrida optimista sin NeuroFuzzy. |
| Arquitectura | LSTM de 50 unidades y salida escalar. `modelsR` y `modelsV` comparten la misma instancia neuronal. |
| Entrada y aprendizaje | `num_time_steps = 1`, `num_features = 1`, Adam, MSE, cinco épocas y lote 20 por llamada a `fit`. |
| Semillas | Sorteo pseudoaleatorio al iniciar cada corrida. No se dispone de la lista de semillas históricas. |
| Resultados | La vinculación de cada tabla de la tesis con las salidas y la configuración efectivas sigue pendiente de corroborar. |

La modificación del número de corridas responde a la aclaración del autor. La revisión se documenta como una adaptación de la copia importada. Las dimensiones de entrada y la construcción del modelo conservan la implementación del archivo de origen. [Código actual](../research/sumoneurofun/main.py).

## Pasos y tiempo simulado

`steps` aumenta en uno después de cada llamada a `traci.simulationStep()` sin argumentos. El control difuso se comprueba cada 300 pasos y el bloque de entrenamiento y predicción se habilita en los múltiplos de 3.000 pasos.

El tiempo simulado transcurrido es el número de pasos multiplicado por la duración de cada paso. `run.sumocfg` y el comando de arranque dejan `step-length` sin fijar. SUMO utiliza un segundo por paso por defecto. La duración efectiva del paso en las corridas originales permanece pendiente de corroborar. [Opciones de tiempo de SUMO](https://sumo.dlr.de/docs/sumo.html#time) · [Llamada de TraCI en SUMO 1.15.0](https://github.com/eclipse-sumo/sumo/blob/v1_15_0/tools/traci/main.py#L185-L194).

La aclaración del autor sitúa los 1.200 s en una corrida optimista sin NeuroFuzzy. Por tanto, esa cifra tiene un alcance distinto del criterio de fin del controlador LSTM distribuido.

## Paquete archivado y registros de otras ejecuciones

El paquete MLP se conserva en la etiqueta `v1.0.1` y en el DOI de esa versión. Su ejecutor, configuración de semillas, dependencias y validación se consultan dentro de ese archivo. [Código v1.0.1](https://github.com/tono88/neurofuzzyTraffic/tree/v1.0.1) · [Zenodo v1.0.1](https://doi.org/10.5281/zenodo.23907073).

El repositorio fuente contiene registros de otras ejecuciones. Cada registro conserva sus propios datos de versión y duración. Para atribuirlos a un resultado concreto de la tesis hace falta vincularlos con las salidas de esa corrida. La declaración de SUMO 1.15.0 recoge la confirmación del autor. [Registro disponible en la fuente histórica](https://github.com/tono88/SUMONeuroFun/blob/57bf0d2629eca652f0238d751742a9a61d35b634/sumo_log.txt).
