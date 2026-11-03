# Validación

## Entorno

Python 3.11, SUMO 1.16.0, TensorFlow 2.15.1, NumPy 1.26.4 y scikit-fuzzy 0.5.0. Las dependencias se definen en `requirements.txt`; `requirements-win64-py311.lock.txt` registra el entorno Windows de las pruebas.

## Comprobaciones

- Integridad SHA-256 de los 42 archivos de `original/`.
- Lista de 100 semillas únicas y concordancia entre TXT y CSV.
- Configuración y dependencias XML del escenario completas y analizables.
- Cinco pruebas automatizadas de métricas, predicciones ausentes, duplicación de semillas y aislamiento de salidas: aprobadas.
- Metadatos de citación validados contra el esquema CFF 1.2.0.
- Dependencias Python: `pip check` sin incompatibilidades declaradas.

## Pruebas de simulación

| Modo | Semilla | Pasos | Resultado |
|---|---:|---:|---|
| Estandar | 179 | 5 | Prueba corta con generación correcta de salidas. |
| Estandar | 179 | 5133 | Corrida completa: 11 176 viajes llegados y cero vehículos pendientes. |
| Fuzzy | 179 | 305 | Registro de 74 725 observaciones. |
| NeuroFuzzy | 179 | 305 | Registro de 47 580 observaciones y 936 pares válidos de predicción. |
| NeuroFuzzy, repetición | 179 | 305 | Datos y métricas idénticos a la primera ejecución. |

La repetición NeuroFuzzy produjo archivos `details.csv`, `prediction_metrics.json` y `traffic_metrics.json` idénticos byte a byte en procesos independientes. Las pruebas de 305 pasos incluyen el entrenamiento inicial y la primera predicción de la MLP.

Los parámetros, versiones y resultados se registran en `evidence.json`. Las pruebas cortas utilizan el estado `truncated`; la corrida completa utiliza `completed`. El diseño experimental de 300 corridas se ejecuta con los comandos del README.
