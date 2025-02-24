import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from adjustText import adjust_text
from util import *
# Load data from xlsx file (provide correct path)
file_path = '../../../resources/240527_source_data/240527_source_data.xlsx'
df = pd.read_excel(file_path, sheet_name='calibrated_co_oxi', skiprows=1)


df['CO2_M'] = pd.to_numeric(df['Au/CeO2-M'], errors='coerce')
df['CO2_R'] = pd.to_numeric(df['Au/CeO2-R'], errors='coerce')
df['CO2_C'] = pd.to_numeric(df['Au/CeO2-C'], errors='coerce')
df = df.dropna(subset=['Temperature_M', 'CO2_M', 'Temperature_R', 'CO2_R', 'Temperature_C', 'CO2_C'])


def plot_CO2_vs_Temperature_with_conversion(df, colors):
    fig = plt.figure(figsize=(6, 6))  # Changed from 6,6 to 5,5
    ax = fig.add_subplot(111)
    plt.plot(df['Temperature_M'], df['CO2_M'], color=colors[0], label='Au/CeO$\mathregular{_2}$ Mace', linewidth=1.5)
    plt.plot(df['Temperature_R'], df['CO2_R'], color=colors[1], label='Au/CeO$\mathregular{_2}$ Rod', linewidth=1.5)
    plt.plot(df['Temperature_C'], df['CO2_C'], color=colors[2], label='Au/CeO$\mathregular{_2}$ Cube', linewidth=1.5)
    # Find temperatures at 5% conversion for each catalyst
    def find_temp_at_conversion(temp_col, conv_col, target=5):
        for i in range(len(df)):
            if df[conv_col].iloc[i] >= target:
                # Linear interpolation
                if i > 0:
                    t1, t2 = df[temp_col].iloc[i-1], df[temp_col].iloc[i]
                    c1, c2 = df[conv_col].iloc[i-1], df[conv_col].iloc[i]
                    return t1 + (t2-t1)*(target-c1)/(c2-c1)
                return df[temp_col].iloc[i]
        return None

    # Add vertical lines and annotations for each catalystß
    for i, (temp_col, conv_col) in enumerate([
        ('Temperature_M', 'CO2_M'),
        ('Temperature_R', 'CO2_R'),
        ('Temperature_C', 'CO2_C')
    ]):
        t5 = find_temp_at_conversion(temp_col, conv_col,target=1)
        if t5 is not None:
            # Add vertical line
            plt.plot([t5, t5], [0, i*7+5], 
                    color=colors[i], linestyle='--', alpha=0.5, linewidth=1)
            # Add temperature annotation
            plt.text(t5, i*7+6, f'{t5:.0f}°C', 
                    ha='center', va='bottom', color=colors[i],
                    fontproperties=font_properties_tick)
    ax.set_position([0.2, 0.2, 0.666, 0.666])  # 4/5 = 0.8

    plt.legend(prop=font_properties_tick, loc='upper left', frameon=False)
    plt.text(80, 5, 'CO Oxidation', fontproperties=font_properties_label)
    plt.xlabel("Temperature (°C)", fontproperties=font_properties_label)
    plt.ylabel("CO Conversion (%)", fontproperties=font_properties_label)
    plt.xticks(fontproperties=font_properties_tick)
    plt.yticks(fontproperties=font_properties_tick)
    plt.ylim(0,100)
    plt.xlim(-100,150)

    # Save the figure
    plt.savefig(f'./output/co_conversion.png', dpi=200)

    # Show the plot
    plt.show()

# Call the function to plot the normalized conversion data
plot_CO2_vs_Temperature_with_conversion(df, colors)

