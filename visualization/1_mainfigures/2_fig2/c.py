import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.append('../../utility')
from util import *
import numpy as np
# Load data from xlsx file
file_path = '../../../resources/240527_source_data/240527_source_data.xlsx'
df = pd.read_excel(file_path, sheet_name='Fig.2b', skiprows=1)

def plot_partial_pressures(df):
    # Separate the data for CO and O2 partial pressures
    co_data = df.iloc[:, :4]  # First 4 columns
    o2_data = df.iloc[:, 4:]  # Last 4 columns

    plt.figure(figsize=(6, 3))

    # Plot CO partial pressure
    plt.subplot(1, 2, 1)
    for i in range(1, len(co_data.columns)):
        x = co_data.iloc[:, 0]
        y = co_data.iloc[:, i]
        coeffs = np.polyfit(np.log(x), np.log(y), 1)  # Linear fit in log-log space
        poly = np.poly1d(coeffs)
        plt.plot(x, np.exp(poly(np.log(x))), label=co_data.columns[i], color=colors[i-1])
        plt.scatter(co_data.iloc[:, 0], co_data.iloc[:, i], s=50, color=colors[i-1])
    plt.xlabel("P$_{CO}$(Kpa)", font_properties=font_properties_label)
    plt.ylabel("Specific activity\n(μmolCO$_{2}$/g$_{ceria}$s$^{-1}$))", font_properties=font_properties_label)
    plt.xscale('log')
    plt.yscale('log')
    plt.ylim(0.25,10)
    plt.xlim(0.5,20)

    plt.xticks(font_properties=font_properties_tick)
    plt.yticks(font_properties=font_properties_tick)
    plt.subplot(1, 2, 2)
    for i in range(1, len(o2_data.columns)):
        x = o2_data.iloc[:, 0]
        y = o2_data.iloc[:, i]
        coeffs = np.polyfit(np.log(x), np.log(y), 1)  # Linear fit in log-log space
        poly = np.poly1d(coeffs)
        plt.plot(x, np.exp(poly(np.log(x))), label=co_data.columns[i], color=colors[i-1])
        plt.scatter(o2_data.iloc[:, 0], o2_data.iloc[:, i], s=50, color=colors[i-1])
    plt.xlabel("P$_{O_{2}}$(Kpa)", font_properties=font_properties_label)
    plt.xscale('log')
    plt.yscale('log')
    plt.ylim(0.25,10)
    plt.xlim(0.5,20)
    plt.xticks(font_properties=font_properties_tick)
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(f'./output/c.png', dpi=200)
    plt.show()
if __name__ == "__main__":
    plot_partial_pressures(df)