# Metodología del código histórico LSTM

El archivo de referencia es [`research/sumoneurofun/main.py`](../research/sumoneurofun/main.py). La revisión 1.0.2 configura 100 corridas de simulación según la indicación del autor. La procedencia del archivo y el alcance de esa modificación se detallan en la [nota de procedencia](PROCEDENCIA_TESIS.md).

## Corridas y semillas

`NUM_SIMULATIONS = 100` determina cuántas veces se inicia SUMO. En cada iteración, el código obtiene una semilla mediante `random.randint(1, 10000)` y la entrega con `--seed`. Los sorteos pueden repetir valores. El código histórico carece de una lista prefijada y de un emparejamiento entre estrategias. Cada llamada a `run()` construye el modelo neuronal de esa corrida.

## Arquitectura y aprendizaje

El modelo contiene `LSTM(50, activation='relu')` y `Dense(1)`. La entrada tiene forma de una observación por una característica. `create_sequences` forma los ejemplos a partir de las observaciones de cola acumuladas para las aproximaciones rojas y verdes.

Las listas `modelsR` y `modelsV` contienen referencias a una misma instancia de `model`. El optimizador es Adam y la pérdida es MSE. Cada llamada a `fit` solicita cinco épocas y lotes de 20 ejemplos. El bloque de entrenamiento y predicción se habilita en los múltiplos de 3.000 pasos, por intersección y grupo. La ejecución satisfactoria de ese bloque y la correspondencia con las tablas de la tesis requieren los registros de las corridas originales.

La partición entre entrenamiento, validación y prueba de los resultados publicados sigue pendiente de documentar. El ajuste LSTM distribuido utiliza las secuencias acumuladas y sus llamadas a `fit` conservan los parámetros del código importado. La definición del modelo emplea Adam y MSE. Las referencias de la tesis a L2, L-BFGS-B o a predictores independientes requieren identificar el procedimiento o archivo correspondiente.

## Control e insumos del escenario

El código calcula la cola a partir de los vehículos detenidos de cada carril y organiza las observaciones según el estado rojo o verde. El bloque de control difuso se comprueba cada 300 pasos de simulación.

`run.sumocfg` referencia la red, la demanda, los autobuses, las paradas, los detectores, los tipos vehiculares y los programas semafóricos `joined*.xml`. Los archivos contienen programas semafóricos estáticos. El programa de referencia efectivamente utilizado para producir las tablas de comparación queda pendiente de vincular con las salidas originales.

## Tiempo y condición de fin

El autor confirma SUMO 1.15.0 con TraCI. El bucle LSTM avanza mientras `getMinExpectedNumber() > 0`. Los 1.200 s, según la aclaración del autor, corresponden a una corrida optimista sin NeuroFuzzy. La duración efectiva del paso debe recuperarse del entorno de las corridas originales. [Relación entre pasos y tiempo](PROCEDENCIA_TESIS.md#pasos-y-tiempo-simulado).

## Salidas conservadas por el programa

El arranque solicita `colaControl{seed}.xml` para las colas y `tripinfo.xml` para los viajes. La configuración también declara salidas de registro y detectores. `lanes_con_semaforos.csv` se genera durante la ejecución.

Algunos nombres de salida se reutilizan entre corridas, incluido `tripinfo.xml`. Una semilla repetida también reutiliza el nombre de su archivo de colas. Este comportamiento proviene del código histórico y debe considerarse al identificar qué archivos respaldan cada tabla de resultados.
