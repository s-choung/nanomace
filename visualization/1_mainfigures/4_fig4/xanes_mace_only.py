import pandas as pd
import matplotlib.pyplot as plt
from util import *


def plot_co_conversion(df,color,title):
    # Create figure with size 5x5 inches
    fig = plt.figure(figsize=(6, 6))  # Changed from 6,6 to 5,5
    ax = fig.add_subplot(111)
    
    # Sample data columns

    sample_labels = ['Ar', 'CO', 'O2', 'CO+O2']  # Simplified labels without LaTeX formatting
    reference_labels = ['Au2O3', 'Au metal']      # Simplified labels without LaTeX formatting
    show_labels=['Ar', 'CO', 'O$\mathregular{_2}$', 'CO+O$\mathregular{_2}$']
    show_reference_labels = ['Au$\mathregular{_2}$O$\mathregular{_3}$', 'Au metal']

    line_styles = [':', '-', '-', '-']
    alpha_values = [0.7, 0.3, 0.6, 1]
    for i, label in enumerate(sample_labels):
        x = df.iloc[:, 0]
        y = df[label]
        ax.plot(x, y, label=show_labels[i], color=color, linestyle=line_styles[i],linewidth=1.5,alpha=alpha_values[i], zorder=2-0.1*i)
    
    # Plot reference data with scatter points
    markers = ['o', 's']
    sizes= [70, 50]
    for i, label in enumerate(reference_labels):
        x = df.iloc[:, 5]
        y = df[label]
        ax.scatter(x, y, 
                   label=show_reference_labels[i],
                   marker=markers[i],
                   s=sizes[i],
                   facecolor='white',
                   edgecolor='gray',
                   alpha=0.5,
                   linewidth=1,
                   zorder=2-0.1*(i+len(sample_labels)))
        #ax.plot(x,y,color='gray',linewidth=1.5,zorder=2-0.1*(i+len(sample_labels)),alpha=0.1)

    # Create inset axes
    axins = ax.inset_axes([0.6, 0.1, 0.30, 0.40])  # [x, y, width, height] in relative coordinates
    
    # Plot the same data in the inset
    for i, label in enumerate(sample_labels):
        x = df.iloc[:, 0]
        y = df[label]
        axins.plot(x, y, color=color, linestyle=line_styles[i],linewidth=1.5,alpha=alpha_values[i])
    
    for i, label in enumerate(reference_labels):
        x = df.iloc[:, 5]
        y = df[label]
        axins.scatter(x, y, marker=markers[i], s=sizes[i], facecolor='white', edgecolor='gray', alpha=0.5, linewidth=1)
        axins.plot(x,y,color='gray',linewidth=1.5,alpha=0.1)

    # Set inset range
    axins.set_xlim(11917, 11921)
    axins.set_ylim(0.7, 1.1)
    
    # Apply tick parameters to inset axes
    axins.set_xticks([11918, 11920])  # Specify exact tick positions
    axins.set_yticks([0.8, 0.9, 1])        # Specify exact tick positions
    for label in axins.get_xticklabels() + axins.get_yticklabels():
        label.set_fontproperties(font_properties_tick)
    
    # Main plot settings
    xmin,xmax= 11900,11940
    ax.set_xlabel("Energy (eV)", fontproperties=font_properties_label)
    ax.set_ylabel("Normalized xμ (E)", fontproperties=font_properties_label)
    ax.set_xticks(range(xmin,xmax+10,10))
    ax.tick_params(labelsize=8)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(font_properties_tick)
    ax.set_xlim(xmin,xmax)
    ax.set_ylim(0,1.2)
    
    # Legend position
    ax.legend(prop=font_properties_tick, loc='upper left', frameon=False)
    
    # Set exact plot position for 4x4 inch plot area in 5x5 inch figure
    # Format is [left, bottom, width, height] in figure coordinates (0 to 1)
    ax.set_position([0.2, 0.2, 0.666, 0.666])  # 4/5 = 0.8
    
    plt.savefig(f'./output/Fig. S23{title}.png', dpi=200)
    plt.show()

if __name__ == "__main__":
    # Load data from xlsx file
    file_path = '../../../resources/240527_source_data/240527_source_data.xlsx'
    df = pd.read_excel(file_path, sheet_name='Fig. S23a', skiprows=1)
    plot_co_conversion(df, colors[0], title='')  # Removed 'a' from title since we're only plotting one figure