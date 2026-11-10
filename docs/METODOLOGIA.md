# Metodología de simulación y control

Esta metodología describe el ejecutor del paquete publicado en la versión 1.0.1. La [nota de procedencia](PROCEDENCIA_TESIS.md) compara ese ejecutor con el `main.py` histórico de la investigación y con el protocolo declarado en la tesis.

## Diseño experimental

El paquete MLP utiliza el escenario Bologna `joined` de SUMO y compara tres condiciones de control: **Estandar**, **Fuzzy** y **NeuroFuzzy**. Su ejecutor reutiliza las semillas de su configuración de referencia entre los modos para una comparación pareada. En el código de la investigación, la semilla se genera de forma pseudoaleatoria al iniciar cada corrida. No se dispone de una lista de esas semillas históricas.

El ejecutor principal es `run_experiment.py`. Los controladores Fuzzy y NeuroFuzzy se implementan en `src/controllers.py`; `original/main.py` contiene su implementación de referencia. El modo Estandar utiliza los planes semafóricos definidos en los archivos del escenario.

## Densidad y control difuso

La densidad se calcula como `100 × número de vehículos / longitud del carril en metros`, en vehículos por 100 m. El universo difuso comprende valores entre 0 y 100.

El sistema estático utiliza los conjuntos trapezoidales bajo `[0,0,30,50]`, medio `[30,50,70,100]` y alto `[70,100,100,100]`. La salida utiliza los conjuntos triangulares actual `[0,0,0.5]` y siguiente `[0.5,1,1]`. Las reglas son: bajo → actual; medio O alto → siguiente. Una salida mayor o igual a 0.5 solicita el cambio de fase.

## Modelo NeuroFuzzy del paquete publicado

Cada semáforo tiene una MLP independiente con entrada de 50 observaciones, una capa Dense de 32 unidades ReLU y una salida lineal. Se utiliza Adam con pérdida MSE. El buffer contiene hasta 200 observaciones y el entrenamiento comienza al superar 100 observaciones, con dos épocas y tamaño de lote 1.

Cada 300 pasos se predice la densidad media del semáforo. Con `p=clip(pred,0,100)`, las funciones de pertenencia se actualizan como bajo `[0,0,p/2,p]`, medio `[p/4,3p/4,p,100]` y alto `[p/2,100,100,100]`.

El callback `EarlyStopping(patience=1)` utiliza el monitor predeterminado `val_loss`. El entrenamiento se realiza con los datos disponibles del buffer y dos épocas. Las predicciones se registran a partir del paso 300.

## Ejecución y registro

| Elemento | Implementación |
|---|---|
| Semillas | El ejecutor MLP aplica la misma semilla a SUMO, Python, NumPy y TensorFlow dentro de cada corrida. |
| Procesos | Un proceso independiente por modo y semilla. |
| Entorno neuronal | CPU, un hilo TensorFlow y operaciones deterministas. |
| Identificación de carriles | NeuroFuzzy conserva la primera aparición de cada carril controlado; Fuzzy recorre la lista de TraCI. |
| Control de fase | Fase actual capturada antes de recorrer los carriles y número de fases de la primera lógica devuelta por TraCI. |
| Observaciones | Escritura incremental en `details.csv`. |
| Predicciones iniciales | Campo vacío hasta la primera predicción neuronal. |
| Umbrales | Columnas `med_lb` y `med_ub`. |
| Salidas SUMO | Configuración, detectores, registros y resultados separados por modo y semilla. |
| Trazabilidad | Manifiestos con parámetros, versiones, huellas SHA-256 y estado de ejecución. |

## Métricas e interpretación

Las métricas de regresión se calculan con pares finitos de observación y predicción. RMSE y MAE se expresan en vehículos por 100 m. Para MAPE, el denominador se reemplaza por 1 cuando el valor real es cero. R² se registra como `null` cuando hay menos de dos pares válidos o la varianza real es nula.

En Fuzzy se registra `y_pred=y_true`: sus errores de regresión y su matriz de confusión describen esa identidad de registro. En NeuroFuzzy se compara la última predicción de densidad media del semáforo con la densidad actual de cada carril, manteniendo la predicción entre actualizaciones de 300 pasos.

La clasificación utiliza Bajo cuando `valor < med_lb`, Medio cuando `med_lb <= valor < med_ub`, y Alto para los valores restantes. Los umbrales de NeuroFuzzy se actualizan con la predicción. La matriz de confusión organiza las filas por clase observada y las columnas por clase predicha, en orden Bajo/Medio/Alto.

La unidad experimental para la comparación de movilidad es la **corrida por semilla**. El resumen de viajes calcula los promedios sobre llegadas y registra los vehículos pendientes en el manifiesto. Las ejecuciones de prueba con `--max-steps` se identifican mediante el estado `truncated`.

## Repetibilidad y entorno

SUMO controla su generador pseudoaleatorio mediante `--seed`. Python, NumPy y TensorFlow utilizan la misma semilla de cada corrida. El ejecutor fija `PYTHONHASHSEED` antes de iniciar el proceso y activa `tf.config.experimental.enable_op_determinism`.

La repetición se realiza con las versiones de software definidas en `requirements.txt` y SUMO 1.16.0. Cada manifiesto registra el entorno concreto. Los comentarios de los XML pueden incluir fecha, rutas y duración real de ejecución; las comparaciones de repetibilidad consideran los datos numéricos y los archivos de métricas.

Referencias: [aleatoriedad en SUMO](https://sumo.dlr.de/docs/Simulation/Randomness.html) y [determinismo en TensorFlow](https://www.tensorflow.org/api_docs/python/tf/config/experimental/enable_op_determinism).
