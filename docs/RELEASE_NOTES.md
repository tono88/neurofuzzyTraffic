# v1.0.1 — Bologna SUMO

Código y escenario del paquete MLP para simulaciones de control semafórico.

- Controladores Fuzzy y NeuroFuzzy con MLP por semáforo, y línea base Estandar.
- Escenario Bologna `joined`: red, demanda, autobuses, paradas, detectores y planes semafóricos.
- Diseño pareado en tres modos de control con la configuración de referencia del ejecutor. Las semillas de las corridas de la investigación se generaron de forma pseudoaleatoria y no se dispone de su lista.
- Semillas de SUMO, Python, NumPy y TensorFlow, con procesos independientes por corrida.
- Registro incremental de observaciones, métricas y manifiestos SHA-256.
- README, metodología, inventario de archivos y pruebas automatizadas.
- Prueba Estandar completa: 11 176 viajes llegados y cero vehículos pendientes.
- Repetición NeuroFuzzy: archivos de datos y métricas idénticos en dos procesos de 305 pasos con semilla 179.
- Código y documentación bajo MIT; componentes SUMO con sus licencias correspondientes.

Repositorio: https://github.com/tono88/neurofuzzyTraffic

DOI: https://doi.org/10.5281/zenodo.23907073
