import xml.etree.ElementTree as ET
import csv

# 1. Parsear el archivo XML
tree = ET.parse('joined_buslanes.net.xml')
root = tree.getroot()
# Simplemente imprimir el nombre del tag del elemento raíz para comprobar si el archivo se ha leído correctamente
print(root.tag)

# 2. Identificar intersecciones con semáforos activos
junctions_with_traffic_lights = set()
for junction in root.findall('junction'):
    if junction.get('type') == 'traffic_light':
        junctions_with_traffic_lights.add(junction.get('id'))

# 3. Listar todas las LANEs conectadas a esas intersecciones
lanes_data = []
for edge in root.findall('edge'):
    # Verificar si el edge está conectado a un junction con semáforo
    if edge.get('to') in junctions_with_traffic_lights:
        for lane in edge.findall('lane'):
            lane_id = lane.get('id')
            lane_length = lane.get('length')
            lanes_data.append((lane_id, lane_length))

# 4. Escribir los datos en un archivo CSV
with open('lanes.csv', 'w', newline='') as csvfile:
    fieldnames = ['Nombre', 'Largo']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for lane in lanes_data:
        writer.writerow({'Nombre': lane[0], 'Largo': lane[1]})
