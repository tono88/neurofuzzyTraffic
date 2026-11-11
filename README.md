# Bologna SUMO — código LSTM de la investigación

**Autor** Estuardo Antonio Sandoval Acevedo · [ORCID 0009-0005-2146-6699](https://orcid.org/0009-0005-2146-6699).

**Dirección de tesis** David Escuin · [ORCID 0000-0003-2062-8092](https://orcid.org/0000-0003-2062-8092).

La rama `main` contiene el código histórico LSTM del escenario Bologna `joined`, con el número de corridas configurado en **100** según la confirmación del autor. El entorno de la investigación confirmado por el autor es **SUMO 1.15.0 con TraCI**. La revisión del software es **1.0.2**. Su DOI de versión es [10.5281/zenodo.24141655](https://doi.org/10.5281/zenodo.24141655).

## Código y parámetros

| Elemento | Implementación disponible |
|---|---|
| Punto de entrada | [`research/sumoneurofun/main.py`](research/sumoneurofun/main.py) y [`run.sumocfg`](research/sumoneurofun/run.sumocfg), con los insumos del escenario en la misma carpeta. |
| Arquitectura | Una capa LSTM de **50 unidades** con activación ReLU y una capa densa de salida de una unidad. |
| Entrada | Una observación de longitud de cola por ejemplo, con `num_time_steps = 1` y `num_features = 1`. |
| Instancias neuronales | Las listas de aproximaciones rojas y verdes referencian una misma instancia del modelo. |
| Entrenamiento | Adam, pérdida MSE, cinco épocas por llamada y tamaño de lote 20. El bloque se habilita en los múltiplos de 3.000 pasos de simulación. |
| Corridas | **100 corridas de simulación**, definidas por `NUM_SIMULATIONS`. La barra de progreso utiliza ese mismo valor. |
| Semillas | Cada corrida genera su semilla pseudoaleatoria mediante `random.randint(1, 10000)` y la pasa a SUMO. No se dispone de la lista de las semillas de las corridas originales. |
| Fin de cada corrida | El bucle continúa mientras SUMO informa vehículos esperados. |

Las 100 corridas corresponden al número de simulaciones. La ventana de entrada y las épocas de entrenamiento son parámetros distintos que se muestran en la tabla. La copia importada solicitaba tres corridas. La revisión 1.0.2 ajusta ese número a 100. [Procedencia y cambios](docs/PROCEDENCIA_TESIS.md).

## Entorno y uso

El autor confirma SUMO **1.15.0** para la investigación. El programa obtiene `sumo` y TraCI de la instalación disponible mediante `SUMO_HOME` y el entorno de ejecución. Las dependencias Python directas se enumeran en [`requirements.txt`](requirements.txt), con sus versiones históricas pendientes de identificar. También se requiere Tkinter para la interfaz de progreso.

El punto de entrada requiere como directorio de trabajo la carpeta que contiene `run.sumocfg`. La invocación documentada desde la raíz del repositorio es:

```text
cd research/sumoneurofun
python main.py
```

El escenario, las rutas de salida y la interfaz se describen en la [metodología del código histórico](docs/METODOLOGIA.md).

## Horizonte y tiempo simulado

El autor aclara que los **1.200 s** corresponden a una corrida optimista **sin el control NeuroFuzzy**. La configuración LSTM distribuida continúa hasta el criterio de fin de vehículos esperados. Su horizonte se interpreta con ese criterio.

El contador `steps` aumenta una vez por llamada a `traci.simulationStep()`. Su conversión a segundos requiere conocer la duración efectiva de cada paso. Ese valor de las corridas originales sigue pendiente de corroborar. [Pasos y protocolo](docs/PROCEDENCIA_TESIS.md#pasos-y-tiempo-simulado).

## Versiones y citación

El paquete MLP con su configuración de comparación entre modos se consulta en la [versión archivada v1.0.1](https://github.com/tono88/neurofuzzyTraffic/tree/v1.0.1) y en su [DOI 10.5281/zenodo.23907073](https://doi.org/10.5281/zenodo.23907073).

Para citar el código LSTM de esta revisión, utilice el [DOI de la versión 1.0.2](https://doi.org/10.5281/zenodo.24141655). El código se identifica en la [etiqueta v1.0.2](https://github.com/tono88/neurofuzzyTraffic/tree/v1.0.2). Los identificadores de cada versión figuran en [`PUBLICATION.json`](PUBLICATION.json) y en la [guía de publicación](docs/PUBLICACION_DOI.md).

## Licencias

El código del proyecto y la documentación se distribuyen bajo [MIT](LICENSE). Los insumos Bologna conservan su licencia y sus avisos de procedencia. [Componentes de terceros](THIRD_PARTY_NOTICES.md).
