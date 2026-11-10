# Componentes de terceros

## Escenario Bologna / iTETRIS

Los archivos `joined*.xml` de `scenarios/bologna/`, `original/` y `research/sumoneurofun/` forman parte del escenario Bologna de SUMO. El escenario público de referencia está en [DLR-TS/sumo-scenarios/bologna](https://github.com/DLR-TS/sumo-scenarios/tree/main/bologna), enlazado desde la [documentación oficial SUMO](https://sumo.dlr.de/docs/Data/Scenarios.html#bologna). La distribución upstream incluye la **Eclipse Public License 2.0**, cuyo texto se conserva en `licenses/SUMO-SCENARIOS-LICENSE.txt`, obtenido de [su archivo LICENSE](https://github.com/DLR-TS/sumo-scenarios/blob/main/LICENSE).

La red indica generación con netconvert `v1_14_1+1311-ab8f708ab1e` en 2022. Los archivos conservan sus comentarios y atributos. La configuración de cada ejecución adapta las rutas y los destinos de salida mediante `make_configuration` en `run_experiment.py`.

Referencia del escenario: Bieker, L., Krajzewicz, D., Morra, A., Michelacci, C. y Cartolano, F. (2014). *Traffic simulation for all: a real world traffic scenario from the city of Bologna*. SUMO2014, pp. 19–26.

## original/randomTrips.py

Herramienta Eclipse SUMO. Copyright © 2010–2023 German Aerospace Center (DLR) y otros, según el encabezado del archivo. El aviso identifica a Daniel Krajzewicz, Jakob Erdmann y Michael Behrisch y declara **SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later**, con la disponibilidad de licencia secundaria sujeta a las condiciones EPL descritas allí. No se elimina ni modifica ese encabezado. El texto EPL de la distribución SUMO 1.16.0 se conserva en `licenses/SUMO-LICENSE.txt`; el propio archivo original enlaza también la [GPL-2.0](https://www.gnu.org/licenses/old-licenses/gpl-2.0-standalone.html).

Esta herramienta se conserva por procedencia; el ejecutor de referencia no la importa ni la ejecuta.

## Dependencias externas

SUMO/TraCI, TensorFlow, NumPy, SciPy, scikit-fuzzy, NetworkX, pandas, scikit-learn, matplotlib y XlsxWriter se instalan separadamente y mantienen sus licencias. No se incluyen sus entornos, paquetes binarios ni instaladores en el repositorio. `requirements.txt` enumera versiones de referencia, no modifica esas licencias.

La licencia MIT de la raíz cubre el código del proyecto y su documentación. No reemplaza estos avisos.
