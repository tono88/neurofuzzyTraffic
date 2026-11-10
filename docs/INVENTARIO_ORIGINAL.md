# Inventario de archivos

Los archivos del directorio `original/` se enumeran en `MANIFEST.json` con su tamaño y su huella SHA-256.

| Archivos | Función |
|---|---|
| `main.py` | MLP por semáforo y controladores Fuzzy y NeuroFuzzy. |
| `main sin metricas.py` | Variante MLP con modos NeuroFuzzy, Fuzzy y Estandar. |
| `mainFuncional.py` | Variante LSTM con control difuso de colas en rojo y verde e interfaz Tkinter. |
| `mainv2.py` | Variante LSTM con barra de progreso y entrenamiento periódico. |
| `mainV3.py` | Variante CNN temporal con entrenamiento periódico. |
| `mainV1.py`, `mainOLD.py` | Versiones de desarrollo de los controladores. |
| `listacalles.py`, `listarCallesSemaforo.py` | Utilidades de inventario de carriles. |
| `ploteo.py`, `ploteo2.py` | Análisis de resultados, gráficas e informes. |
| `probador.py` | Utilidad de exploración y desnormalización de datos. |
| `randomTrips.py` | Herramienta SUMO de generación de rutas, con licencia de terceros. |
| `run.sumocfg`, `joined*.xml` | Configuración, red, demanda, paradas, detectores, tipos vehiculares y planes semafóricos. |
| `neurofuzzy_model.h5`, `scaler_y.pkl` | Modelo neuronal y objeto de escalado. El ejecutor principal entrena la MLP durante cada corrida. |
| `tripinfos.xml`, `e1_output.xml`, `sumo_log.txt` | Archivos de resultados y registro SUMO. |
| `resultados_metricas.xlsx` | Hoja de métricas de simulación. |
| `Figure_*.png`, `*confusion*.png`, `mapa.jpg` | Gráficas, matrices de confusión y mapa. |
| `lane_lengths.csv`, `lanes.csv`, `lanes_con_semaforos.csv`, `resumen_calles.csv` | Inventarios de carriles y resúmenes. |

La ejecución reproducible utiliza `run_experiment.py`, `src/controllers.py`, `config/seeds.txt` y los archivos de `scenarios/bologna/`.

La importación del `main.py` de `SUMONeuroFun` y sus insumos se conserva en `research/sumoneurofun/`. Su origen y sus diferencias con los archivos de `original/` figuran en la [nota de procedencia](PROCEDENCIA_TESIS.md).
