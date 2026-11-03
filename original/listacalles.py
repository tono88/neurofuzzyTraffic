import os
import csv
import traci
import traci.constants as tc

def extract_lane_lengths(sumo_cmd, sumo_cfg_file):
    # Iniciar SUMO con TRACI
    traci.start([sumo_cmd, "-c", sumo_cfg_file])

    lanes = traci.lane.getIDList()  # Obtener lista de todos los lanes
    lane_data = []

    for lane_id in lanes:
        length = traci.lane.getLength(lane_id)  # Obtener el largo de cada lane
        lane_data.append([lane_id, length])

    traci.close()
    return lane_data

def write_to_csv(lane_data, output_filename="lane_lengths.csv"):
    with open(output_filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["Nombre", "Largo"])  # Escribir encabezados
        for item in lane_data:
            csv_writer.writerow(item)  # Escribir datos de cada lane

if __name__ == "__main__":
    # Definir el comando de SUMO y el archivo de configuración
    SUMO_CMD = "sumo"
    SUMO_CFG = "run.sumocfg"
    
    lane_data = extract_lane_lengths(SUMO_CMD, SUMO_CFG)
    write_to_csv(lane_data)

