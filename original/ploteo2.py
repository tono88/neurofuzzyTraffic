import os
import tempfile
import xml.etree.ElementTree as ET
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
from scipy.stats import sem, t
from collections import defaultdict
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
        std_devs = np.std(data_np[:, 1:], axis=0)
        conf_intervals = [confidence_interval(data_np[:, i + 1]) for i in range(3)]
        
        stats_data[lane_id] = {
            "mean": means,
            "std_dev": std_devs,
            "confidence_interval": conf_intervals
        }
        
    return stats_data

def confidence_interval(data, confidence=0.95):
    n = len(data)
    m = np.mean(data)
    std_err = sem(data)
    h = std_err * t.ppf((1 + confidence) / 2, n - 1)

    return m - h, m + h

def plot_distribution(lane_data, stats_data, folder):
    file_paths = []
    for lane_id, data in lane_data.items():
        data_np = np.array(data)
        
        fig, ax = plt.subplots(3, 1, figsize=(10, 15))
        ax[0].set_title(f"Lane {lane_id} - Queueing Time Distribution")
        ax[1].set_title(f"Lane {lane_id} - Queueing Length Distribution")
        ax[2].set_title(f"Lane {lane_id} - Queueing Length Experimental Distribution")
        
        for i in range(3):
            sns.histplot(data_np[:, i + 1], kde=True, ax=ax[i], label="Data")
            ax[i].axvline(stats_data[lane_id]["mean"][i], color="r", linestyle="--", label="Mean")  # Change this line
            ax[i].axvspan(stats_data[lane_id]["confidence_interval"][i][0], stats_data[lane_id]["confidence_interval"][i][1], color="gray", alpha=0.3, label="95% CI")
            ax[i].legend()
        
        
        file_path = os.path.join(folder, f"distribution_plot_{lane_id}.png")
        plt.savefig(file_path)
        plt.close(fig)  # Close the figure after saving it
        file_paths.append(file_path)
    return file_paths


def plot_violin(lane_data,folder):
    lane_ids = sorted(list(lane_data.keys()))
    num_lanes = len(lane_ids)
    file_paths = []
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

        file_path = os.path.join(folder, f"violin_plot_{lane_id}.png")
        plt.savefig(file_path)
        plt.close(fig)  # Close the figure after saving it
        file_paths.append(file_path)
    return file_paths




def create_pdf_report(lane_data, stats_data, violin_plot_paths, distribution_plot_paths, output_file):
    doc = SimpleDocTemplate(output_file, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    title = Paragraph("Density Report", styles["h1"])
    story.append(title)
    story.append(Spacer(1, 12))

    for lane_id in lane_data:
        lane_title = Paragraph(f"Lane {lane_id}", styles["h2"])
        story.append(lane_title)
        story.append(Spacer(1, 12))

        mean = stats_data[lane_id]["mean"]
        std_dev = stats_data[lane_id]["std_dev"]
        ci_low, ci_high = stats_data[lane_id]["confidence_interval"]

        for i in range(3):
            mean_paragraph = Paragraph(f"Mean {['Queueing Time', 'Queueing Length', 'Queueing Length Experimental'][i]}: {mean[i]:.2f}", styles["BodyText"])
            std_dev_paragraph = Paragraph(f"Standard Deviation {['Queueing Time', 'Queueing Length', 'Queueing Length Experimental'][i]}: {std_dev[i]:.2f}", styles["BodyText"])
            ci_paragraph = Paragraph(f"95% Confidence Interval {['Queueing Time', 'Queueing Length', 'Queueing Length Experimental'][i]}: [{ci_low[i]:.2f}, {ci_high[i]:.2f}]", styles["BodyText"])

            story.extend([mean_paragraph, std_dev_paragraph, ci_paragraph, Spacer(1, 12)])

        violin_plot_path = violin_plot_paths[lane_id]
        distribution_plot_path = distribution_plot_paths[lane_id]

        violin_plot_image = Image(violin_plot_path, width=300, height=300)
        distribution_plot_image = Image(distribution_plot_path, width=300, height=300)

        story.extend([violin_plot_image, Spacer(1, 12), distribution_plot_image, Spacer(1, 24)])

    doc.build(story)


def main():
    file_name = "colaControl5346.xml"
    output_file = "colaControl5346.pdf"

    lane_data = parse_xml_file(file_name)
    stats_data = calculate_statistics(lane_data)
    #print("Lane data:", lane_data)


    with tempfile.TemporaryDirectory() as temp_dir:
        violin_plot_paths = plot_violin(lane_data, temp_dir)
        distribution_plot_paths = plot_distribution(lane_data, stats_data, temp_dir)

        violin_plot_paths_dict = {lane_id: path for lane_id, path in zip(lane_data.keys(), violin_plot_paths)}
        distribution_plot_paths_dict = {lane_id: path for lane_id, path in zip(lane_data.keys(), distribution_plot_paths)}

        create_pdf_report(lane_data, stats_data, violin_plot_paths_dict, distribution_plot_paths_dict, output_file)

if __name__ == "__main__":
    main()

