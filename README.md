# Bologna SUMO — código de la tesis y paquete de simulación

**Autor** Estuardo Antonio Sandoval Acevedo · [ORCID 0009-0005-2146-6699](https://orcid.org/0009-0005-2146-6699).

**Dirección de tesis** David Escuin · [ORCID 0000-0003-2062-8092](https://orcid.org/0000-0003-2062-8092).

## Qué código corresponde a la tesis

| Implementación | Punto de entrada y arquitectura |
|---|---|
| **LSTM descrita en la tesis** | [`research/sumoneurofun/main.py`](research/sumoneurofun/main.py#L417-L425) define una **LSTM de 50 unidades** seguida de una salida escalar. Procesa un paso temporal de una característica de longitud de cola. |
| **MLP del paquete 1.0.1 archivado en Zenodo** | [`run_experiment.py`](run_experiment.py) usa [`src/controllers.py`](src/controllers.py#L79-L89) para el modo NeuroFuzzy. Define una **MLP de 32 unidades ReLU**, con **50 observaciones de densidad media** como entrada y una salida escalar. |

El autor identificó el `main.py` de SUMONeuroFun como el código de la investigación. La copia pública conserva ese archivo y sus insumos tal como estaban en el commit de origen. El número 50 designa las unidades recurrentes en la LSTM y la longitud de la ventana de entrada en la MLP. [Nota de procedencia y comparación técnica](docs/PROCEDENCIA_TESIS.md).

### Consultar el código LSTM

El archivo [`main.py`](research/sumoneurofun/main.py) abre `run.sumocfg` mediante una ruta relativa. La [carpeta del código de la tesis](research/sumoneurofun/) reúne ambos archivos y los insumos del escenario Bologna `joined`. El programa genera una semilla pseudoaleatoria por corrida y la pasa a SUMO con `--seed`. No se dispone de una lista de las semillas de las corridas de la investigación.

La tesis declara **SUMO 1.15 con TraCI** y **1.200 s por escenario**. La configuración importada carece de un tiempo final explícito. Un registro de la rama fuente documenta una corrida distinta en duración, de **5.159 s** con **SUMO 1.16.0**, cuyo vínculo con las tablas originales está sin confirmar. La reproducción exacta requiere identificar el protocolo efectivo de las corridas que produjeron esas tablas. [Comparación del protocolo declarado con la evidencia disponible](docs/PROCEDENCIA_TESIS.md#versión-de-sumo-y-horizonte).

El código programa el primer bloque de entrenamiento de la LSTM para el **paso de simulación 3.000**. Para compararlo con el horizonte de **1.200 s**, hay que convertir los pasos a tiempo simulado usando la duración efectiva de cada paso. [Pasos de simulación y segundos simulados](docs/PROCEDENCIA_TESIS.md#pasos-de-simulación-y-segundos-simulados).

### Consultar el paquete MLP

El paquete 1.0.1 compara los modos Estandar, Fuzzy y NeuroFuzzy. Sus requisitos, comandos, salidas y validación están en la [guía del ejecutor MLP](docs/GUIA_PAQUETE_MLP.md).

## Citación y archivo

La versión **1.0.1** está archivada y publicada en [Zenodo](https://zenodo.org/records/23907073). Su DOI de versión es **[10.5281/zenodo.23907073](https://doi.org/10.5281/zenodo.23907073)**. Los metadatos de citación están en `CITATION.cff` y `.zenodo.json`.

Para citar el código LSTM, indique el archivo y el commit de GitHub consultado. Para citar el paquete MLP 1.0.1, utilice su DOI de versión. El DOI de concepto [10.5281/zenodo.23907072](https://doi.org/10.5281/zenodo.23907072) agrupa las versiones del software. Los enlaces del repositorio y del depósito están registrados en [PUBLICATION.json](PUBLICATION.json).

## Licencias y procedencia

El código del proyecto y su documentación se distribuyen bajo [MIT](LICENSE). Los archivos de terceros mantienen sus condiciones: escenario SUMO bajo los términos de su distribución de origen y `original/randomTrips.py` con su aviso **EPL-2.0 OR GPL-2.0-or-later**. Se incluyen los textos oficiales disponibles en `licenses/` y el detalle en [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). La licencia MIT no cambia los derechos de los componentes de terceros ni los de las dependencias instaladas por separado.

## Referencias del escenario y del entorno

- Bieker, L., Krajzewicz, D., Morra, A., Michelacci, C. y Cartolano, F. (2014). *Traffic simulation for all: a real world traffic scenario from the city of Bologna*. SUMO2014, pp. 19–26. [Registro DLR y escenario](https://sumo.dlr.de/docs/Data/Scenarios.html#bologna).
- [SUMO: control de aleatoriedad mediante `--seed`](https://sumo.dlr.de/docs/Simulation/Randomness.html).
- [TensorFlow: determinismo de operaciones](https://www.tensorflow.org/api_docs/python/tf/config/experimental/enable_op_determinism).
- [Zenodo: habilitar el repositorio](https://help.zenodo.org/docs/github/enable-repository/) y [archivar una release](https://help.zenodo.org/docs/github/archive-software/github-upload/).
