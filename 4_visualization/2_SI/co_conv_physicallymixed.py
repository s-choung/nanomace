import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from adjustText import adjust_text
from util import *
# Load data from xlsx file (provide correct path)
file_path = '../../resources/240527_source_data/240527_source_data.xlsx'
df = pd.read_excel(file_path, sheet_name='calib_Au_R_C', skiprows=1)


df['CO2_M'] = pd.to_numeric(df['Au/CeO2-M'], errors='coerce')
df['CO2_R'] = pd.to_numeric(df['Au/CeO2-R'], errors='coerce')
df['CO2_C'] = pd.to_numeric(df['Au/CeO2-C'], errors='coerce')
df['CO2_R_C'] = pd.to_numeric(df['Au/CeO2-R_C'], errors='coerce')
df = df.dropna(subset=['Temperature_M', 'CO2_M', 'Temperature_R', 'CO2_R', 'Temperature_C', 'CO2_C','Temperature_R_C','CO2_R_C'])


def plot_CO2_vs_Temperature_with_conversion(df, colors):
    plt.figure(figsize=(5, 5))
    plt.plot(df['Temperature_M'], df['CO2_M'], color=colors[0],label='Au/CeO$\mathregular{_2}$ Mace', linewidth=1.5)
    plt.plot(df['Temperature_R'], df['CO2_R'], color=colors[1], label='Au/CeO$\mathregular{_2}$ Rod', linewidth=1.5)
    plt.plot(df['Temperature_C'], df['CO2_C'], color=colors[2], label='Au/CeO$\mathregular{_2}$ Cube', linewidth=1.5)
    plt.plot(df['Temperature_R_C'], df['CO2_R_C'], color='gray', label='Physically mixed', linewidth=1.5)
    plt.xlabel("Temperature (°C)", fontproperties=font_properties_label)
    plt.ylabel("CO Conversion (%)", fontproperties=font_properties_label)
    plt.xticks(fontproperties=font_properties_tick)
    plt.yticks(fontproperties=font_properties_tick)
    plt.ylim(0,100)
    plt.xlim(-50,100)
    plt.legend(loc='upper left', prop=font_properties_label)
    # Save the figure
    plt.savefig(f'./output/test.png', dpi=200)

    # Show the plot
    plt.show()

# Call the function to plot the normalized conversion data
plot_CO2_vs_Temperature_with_conversion(df, colors)

