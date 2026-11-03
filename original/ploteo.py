import xml.etree.ElementTree as ET
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

def parse_xml_file(file_name):
    tree = ET.parse(file_name)
    root = tree.getroot()
    
    lane_data = {}
    
    for data in root.findall("data"):
        timestep = float(data.get("timestep"))
        
        for lane in data.find("lanes"):
            lane_id = lane.get("id")
            queueing_time = float(lane.get("queueing_time"))
            queueing_length = float(lane.get("queueing_length"))
            queueing_length_experimental = float(lane.get("queueing_length_experimental"))
            
            if lane_id not in lane_data:
                lane_data[lane_id] = []
                
            lane_data[lane_id].append((timestep, queueing_time, queueing_length, queueing_length_experimental))
            
    return lane_data

def calculate_statistics(lane_data):
    stats_data = {}
    
    for lane_id, data in lane_data.items():
        data_np = np.array(data)
        means = np.mean(data_np[:, 1:], axis=0)
        conf_intervals = [stats.t.interval(0.95, len(data_np) - 1, loc=mean, scale=stats.sem(data_np[:, i + 1])) for i, mean in enumerate(means)]
        
        stats_data[lane_id] = (means, conf_intervals)
        
    return stats_data

def plot_data(lane_data, stats_data):
    for lane_id, data in lane_data.items():
        data_np = np.array(data)
        
        fig, ax = plt.subplots(3, 1, figsize=(10, 15))
        ax[0].set_title(f"Lane {lane_id} - Queueing Time")
        ax[1].set_title(f"Lane {lane_id} - Queueing Length")
        ax[2].set_title(f"Lane {lane_id} - Queueing Length Experimental")
        
        for i in range(3):
            ax[i].plot(data_np[:, 0], data_np[:, i + 1], label="Data")
            ax[i].axhline(stats_data[lane_id][0][i], color="r", linestyle="--", label="Mean")
            ax[i].fill_between(data_np[:, 0], stats_data[lane_id][1][i][0], stats_data[lane_id][1][i][1], color="gray", alpha=0.3, label="95% CI")
            ax[i].legend()
        
        plt.show()
import seaborn as sns

def plot_distribution(lane_data, stats_data):
    for lane_id, data in lane_data.items():
        data_np = np.array(data)
        
        fig, ax = plt.subplots(3, 1, figsize=(10, 15))
        ax[0].set_title(f"Lane {lane_id} - Queueing Time Distribution")
        ax[1].set_title(f"Lane {lane_id} - Queueing Length Distribution")
        ax[2].set_title(f"Lane {lane_id} - Queueing Length Experimental Distribution")
        
        for i in range(3):
            sns.histplot(data_np[:, i + 1], kde=True, ax=ax[i], label="Data")
            ax[i].axvline(stats_data[lane_id][0][i], color="r", linestyle="--", label="Mean")
            ax[i].axvspan(stats_data[lane_id][1][i][0], stats_data[lane_id][1][i][1], color="gray", alpha=0.3, label="95% CI")
            ax[i].legend()
        
        plt.show()


def plot_violin(lane_data):
    lane_ids = sorted(list(lane_data.keys()))
    num_lanes = len(lane_ids)
    all_data = [[[], [], []] for _ in range(3)]

    for lane_id in lane_ids:
        data_np = np.array(lane_data[lane_id])
        for j in range(3):
            all_data[j].append(data_np[:, j + 1].tolist())

    for i, lane_id in enumerate(lane_ids):
        fig, ax = plt.subplots(3, 1, figsize=(5, 15))
        ax[0].set_title(f"Queueing Time Violin Plot for Lane {lane_id}")
        ax[1].set_title(f"Queueing Length Violin Plot for Lane {lane_id}")
        ax[2].set_title(f"Queueing Length Experimental Violin Plot for Lane {lane_id}")

        for j in range(3):
            filtered_data = all_data[j][i]
            if filtered_data:
                violin_parts = ax[j].violinplot([filtered_data], showmeans=True, showmedians=False, showextrema=False)
                for pc in violin_parts['bodies']:
                    pc.set_facecolor('#D43F3A')
                    pc.set_edgecolor('black')
                    pc.set_alpha(0.7)

                ax[j].scatter(1, np.mean(filtered_data), color="blue", label="Mean", zorder=3)
                ax[j].set_xticks([1])
                ax[j].set_xticklabels([lane_id])
                ax[j].legend()

        plt.show()




if __name__ == "__main__":
    file_name = "colaControl5346.xml"
    lane_data = parse_xml_file(file_name)
    stats_data = calculate_statistics(lane_data)
    #plot_data(lane_data, stats_data)
    #plot_distribution(lane_data, stats_data)
    plot_violin(lane_data)
