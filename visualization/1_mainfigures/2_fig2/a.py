import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.append('../../utility')
from util import *
# Load data from xlsx file
file_path = '../../../resources/240527_source_data/240527_source_data.xlsx'
df = pd.read_excel(file_path, sheet_name='Fig. S7', skiprows=1)
new_colors=['#840032','#e59500','#002642','gray']
def plot_co_conversion(df):
    plt.figure(figsize=(6, 5))
    labels = df.columns[1:] 
    for i, label in enumerate(labels):
        x = df["Temperature (°C)"]
        y = df[label]
        plt.plot(x, y, label=label, color=new_colors[i], linewidth=1.5, zorder=2-0.1*i)
        plt.scatter(x, y, color=new_colors[i], s=100, marker='o', linewidth=1.5, zorder=2-0.1*i)
    plt.xlabel("Temperature (°C)", fontproperties=font_properties_label)
    plt.ylabel("CO Conversion (%)", fontproperties=font_properties_label)
    plt.xticks(fontproperties=font_properties_tick)
    plt.yticks(fontproperties=font_properties_tick)
    plt.xlim(50,500)
    plt.tight_layout()
    plt.savefig(f'./output/fig2_a.png', dpi=200)
    plt.show()

if __name__ == "__main__":
    plot_co_conversion(df)