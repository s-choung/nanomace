import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from util import *
# Load data from xlsx file (provide correct path)
file_path = '../../../resources/mace_exp_raw_data/GC/WGSstability/240621_GC.xlsx'
df = pd.read_excel(file_path,  skiprows=3)

# First, let's print the column names to see what we're working with
name_map = {'ANM': 'Au/CeO$\mathregular{_2}$ Mace', 'ANR': 'Au/CeO$\mathregular{_2}$ Rod', 'ANC': 'Au/CeO$\mathregular{_2}$ Cube'}

def plot_stability_conversion(df):
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)    
    # Use the conversion columns (columns H, I, J from Excel)
    labels = ['ANM', 'ANR', 'ANC']
    conversion_columns = [7, 8, 9]  # These correspond to the conversion columns
    
    for i, (label, col_idx) in enumerate(zip(labels, conversion_columns)):
        x = df.iloc[:, 0]  # Time column
        y = df.iloc[:, col_idx] *100 # Values are already numeric
        #plt.plot(x, y, label=name_map[label], color=colors[i], linewidth=1.5, zorder=2-0.1*i)
        # Plot scatter points every 2 hours
        plt.scatter(x, y, color=colors[i], label=name_map[label], s=50, marker='o', linewidth=1.5, zorder=2-0.1*i)
    
    plt.xlabel("Time (h)", fontproperties=font_properties_label)
    plt.ylabel("CO Conversion (%)", fontproperties=font_properties_label)
    plt.xticks(fontproperties=font_properties_tick)
    plt.yticks(fontproperties=font_properties_tick)
    plt.tight_layout()
    plt.ylim(0, 100)
    plt.xlim(0, 80)
    plt.legend(prop=font_properties_tick, loc='upper left', frameon=False)

    ax.set_position([0.2, 0.2, 0.666, 0.666])
   
    plt.savefig('./output/wgs_stability.png', dpi=200)




plot_stability_conversion(df)