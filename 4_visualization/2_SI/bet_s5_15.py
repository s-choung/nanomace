import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from util import *
# Load data from xlsx file (provide correct path)


def plot_BET(df,output_filename,label_names):
    plt.figure(figsize=(5, 4))
    labels =df.columns[[1,2,3]]
    x_labels = label_names  # Added this line

    for i, label in enumerate(labels):
        x = i
        y = df[label].values[0]
        plt.bar(x, y, label=label, color=colors[i], width=0.5, zorder=2-0.1*i)
        plt.text(x, y+1, f'{y:.2f}', ha='center', va='bottom', fontproperties=font_properties_tick)
    plt.ylabel(r'$\mathregular{S_A (m^2/g)}$', fontproperties=font_properties_label)
    plt.xticks(range(len(x_labels)), x_labels, fontproperties=font_properties_tick)
    plt.yticks(fontproperties=font_properties_tick)
    plt.xlim(-0.5,2.5)
    plt.ylim(0,70)
    plt.tight_layout()
    plt.savefig(f'./output/{output_filename}.png', dpi=200)
    plt.show()


file_path = '../../resources/240527_source_data/240527_source_data.xlsx'
df = pd.read_excel(file_path, sheet_name='Fig. S3', skiprows=1)
plot_BET(df,'bet_s5',['Nanomace', 'Nanorod', 'Nanocube'])
df2 = pd.read_excel(file_path, sheet_name='Fig. S15', skiprows=1)
plot_BET(df2,'bet_s15',['Au/Nanomace', 'Au/Nanorod', 'Au/Nanocube'])