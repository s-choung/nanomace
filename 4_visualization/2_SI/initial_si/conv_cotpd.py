import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from util import *
# Load data from xlsx file (provide correct path)
file_path = './240527_source_data.xlsx'
df = pd.read_excel(file_path, sheet_name=0, skiprows=1)
print(df)
def plot_co_conversion(df):
    plt.figure(figsize=(5, 4))
    labels = df.columns[1:] 
    for i, label in enumerate(labels):
        x = df["Temperature (°C)"]
        y = df[label]
        plt.plot(x, y, label=label, color=colors[i], linewidth=1.5, zorder=2-0.1*i)
        plt.scatter(x, y, color=colors[i], s=100, marker='o', linewidth=1.5, zorder=2-0.1*i)
    plt.xlabel("Temperature (°C)", fontproperties=font_properties_label)
    plt.ylabel("CO Conversion (%)", fontproperties=font_properties_label)
    plt.xticks(fontproperties=font_properties_tick)
    plt.yticks(fontproperties=font_properties_tick)
    plt.tight_layout()
    
    plt.savefig(f'./1_pngs/fig2_a.png', dpi=200)
    plt.show()

plot_co_conversion(df)


# Load data from the Excel file
df = pd.read_excel(file_path, sheet_name='co tpd')
print(df)
def plot_sample_data(df):
    plt.figure(figsize=(5, 4))  # Adjusted for better visibility
    temperature = df.columns[0]  # Typically the first column for temperature
    labels = ['M','R','C']  # The remaining columns are assumed to be data columns

    for i, label in enumerate(labels):
        x = df[temperature]
        y = df[label]
        plt.plot(x, y, label=label, color=colors[i % len(colors)], linewidth=2.5)

    plt.xlabel("Temperature (°C)", fontproperties=font_properties_label)
    plt.ylabel("CO$_{2}$ MS signal m/z=44", fontproperties=font_properties_label)
    plt.xticks(fontproperties=font_properties_tick)
    plt.yticks([])
    plt.xlim(50,800)
    plt.tight_layout()
    plt.savefig(f'./1_pngs/fig2_tpr.png', dpi=200)

    plt.show()


plot_sample_data(df)
