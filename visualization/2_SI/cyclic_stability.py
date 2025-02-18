import pandas as pd
import matplotlib.pyplot as plt
from util import *
import glob
import os
from matplotlib.lines import Line2D
import numpy as np
# Define the data directory
data_dir = '../resources/mace_exp_raw_data/LT_CO_oxidation/cycle_data/'

def read_csv_data(file_path,mz=44):
    # Try different encodings
    try:
        df = pd.read_csv(file_path, skiprows=6, encoding='utf-8')
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(file_path, skiprows=6, encoding='cp949')
        except UnicodeDecodeError:
            df = pd.read_csv(file_path, skiprows=6, encoding='euc-kr')
    
    # Find the temperature column (any column containing 'temp' and 'cell')
    temp_col = [col for col in df.columns if 'temp' in col.lower() and 'cell' in col.lower()][0]
    
    # Filter the DataFrame to start after 'Time(s)' > 600
    df = df[df['Time(s)'] > 600]
    
    # Extract only the temperature and m/z=44 columns
    return df[[temp_col, f'm/z={mz}(A)']]

def extract_cycle_label(filename):
    # Extract cycle label from filename
    filename = os.path.basename(filename)
    if '1cycle' in filename.lower() or '1st' in filename.lower():
        return '1st cycle'
    elif '2cycle' in filename.lower() or '2nd' in filename.lower():
        return '2nd cycle'
    elif '3cycle' in filename.lower() or '3rd' in filename.lower():
        return '3rd cycle'
    elif '4th' in filename.lower():
        return '4th cycle'
    elif '5th' in filename.lower():
        return '5th cycle'
    else:
        return filename
def get_cycle_number(filename):
    filename = filename.lower()
    if '1cycle' in filename or '1st' in filename:
        return 1
    elif '2cycle' in filename or '2nd' in filename:
        return 2
    elif '3rd' in filename: #'3cycle' in filename or 
        return 3
    elif '4th' in filename:
        return 4
    elif '5th' in filename:
        return 5
    return 999  # for any unmatched files, put them at the end
def plot_cyclic_stability(mz=44):
    # Get only old files (removing new files logic)
    all_files = glob.glob(os.path.join(data_dir, '*CeO2-M*.CSV'))
    files = [f for f in all_files if 'new' not in f.lower() and '3cycle' not in f.lower()]
    # Define sorting key function

    # Sort files based on cycle number
    files.sort(key=get_cycle_number)
    
    # Replace the single subplot creation with GridSpec layout
    fig = plt.figure(figsize=(8, 4))
    gs = fig.add_gridspec(2, 2, width_ratios=[2, 1], height_ratios=[1, 1])
    
    # Create three axes
    ax = fig.add_subplot(gs[:, 0])    # Main plot (left side)
    ax2 = fig.add_subplot(gs[0, 1])   # Upper right
    ax3 = fig.add_subplot(gs[1, 1])   # Lower right
    
    # Create colormap
    cmap = plt.cm.inferno  # You can change to other cmaps like 'plasma', 'magma', etc.
    colors = [cmap(i/len(files)) for i in range(len(files))]
    
    # Filter and plot files
    valid_files = []
    for file in files:
        df = read_csv_data(file,mz)
        temp_col = [col for col in df.columns if 'temp' in col.lower() and 'cell' in col.lower()][0]
        
        # Check if data extends beyond 130°C
        if df[temp_col].max() > 130:
            valid_files.append(file)
    
    # Create new color map based on valid files count
    colors = [cmap(i/len(valid_files)) for i in range(len(valid_files))]
    
    legend_elements = []

    all_y_values = pd.concat([read_csv_data(f,mz)[f'm/z={mz}(A)'] for f in valid_files])
    y_max = all_y_values.max() 
    print(y_max)
    for i, file in enumerate(valid_files):
        df = read_csv_data(file,mz)
        temp_col = [col for col in df.columns if 'temp' in col.lower() and 'cell' in col.lower()][0]
        # Extract cycle number from filename
        filename = os.path.basename(file)
        if '1cycle' in filename.lower() or '1st' in filename.lower():
            cycle_label = '1st'
        elif '2cycle' in filename.lower() or '2nd' in filename.lower():
            cycle_label = '2nd'
        elif '3cycle' in filename.lower() or '3rd' in filename.lower():
            cycle_label = '3rd'
        elif '4th' in filename.lower():
            cycle_label = '4th'
        elif '5th' in filename.lower():
            cycle_label = '5th'
        else:
            cycle_label = filename
        
        legend_elements.append(Line2D([0], [0], color=colors[i], marker='o',   label=cycle_label, markersize=8,  linewidth=1.5, alpha=0.7))
        calib= 4.8E-011
        term = 2
        #y_max = 100
        ax3.set_xlim(-100, -40)
        ax2.set_xlim(60, 120)
        for j, curr_ax in enumerate([ax, ax2, ax3]):
            y_data = (df[f'm/z={mz}(A)']/calib)*100
            #y_data = np.array(df[f'm/z={mz}(A)'])/y_max * 100
            curr_ax.plot(df[temp_col][::term], y_data[::term], label=cycle_label, color=colors[i], alpha=0.7, linewidth=1.5, zorder=1+j)
            curr_ax.scatter(df[temp_col][::term], y_data[::term], label=cycle_label, color=colors[i], alpha=0.7, s=10, marker='o', zorder=1+j )
    if mz == 44:
        ylabel = "Normalized CO$_2$ concentration (%)"
        ax3.set_ylim(0.0, 5)
        ax2.set_ylim(65, 80)
    else:
        ylabel = "Normalized CO concentration (%)"
        ax3.set_ylim(75, 90)
        ax2.set_ylim(20, 35)
    ax.set_ylim(0, 100)
    ax.set_xlim(-100, 150)
    ax.set_xlabel("Temperature (°C)", fontproperties=font_properties_label)
    ax.set_ylabel(ylabel, fontproperties=font_properties_label)
    
    ax2.set_xlabel("Temperature (°C)", fontproperties=font_properties_label)
    ax3.set_xlabel("Temperature (°C)", fontproperties=font_properties_label)
    
    # Set font properties for all axes
    for curr_ax in [ax, ax2, ax3]:
        for tick in curr_ax.get_xticklabels() + curr_ax.get_yticklabels():
            tick.set_fontproperties(font_properties_label)
    
    # Set specific ticks and labels for ax2 and ax3
    for curr_ax in [ax2, ax3]:
        # Set y-ticks at intervals of 5
        curr_ax.set_yticks(np.arange(curr_ax.get_ylim()[0], curr_ax.get_ylim()[1] + 1, 5))
        # Set y-tick labels rounded to 2 decimal places
        curr_ax.set_yticklabels([f"{int(tick)}" for tick in curr_ax.get_yticks()])
    
    # Add legend only to main plot
    ax.legend(handles=legend_elements, prop=font_properties_label)
    
    plt.tight_layout()
    plt.savefig(f'./1_pngs/cyclic_stability_mace_{mz}.png', dpi=200, bbox_inches='tight')
    #plt.show()


def plot_merge_cyclic_stability():
    fig, axes = plt.subplots(2, 1, figsize=(8, 8))  # Create a figure with two subplots side by side

    # Load images
    ax1_img = plt.imread('./1_pngs/cyclic_stability_mace_44.png')
    ax2_img = plt.imread('./1_pngs/cyclic_stability_mace_28.png')

    # Display images
    axes[0].imshow(ax1_img)
    axes[1].imshow(ax2_img)

    # Add text with higher z-order using font_properties_label
    axes[0].text(0.0, 0.99, 'a', transform=axes[0].transAxes, fontsize=16, fontweight='bold', color='black', zorder=10, va='top')
    axes[1].text(0.0, 0.99, 'b', transform=axes[1].transAxes, fontsize=16, fontweight='bold', color='black', zorder=10, va='top')

    # Remove axes for a cleaner look
    for ax in axes:
        ax.axis('off')

    plt.tight_layout()
    plt.savefig('./1_pngs/merged_cyclic_stability.png', dpi=200, bbox_inches='tight')
    plt.show()
if __name__ == "__main__":
    plot_cyclic_stability(44)
    plot_cyclic_stability(28)
    plot_merge_cyclic_stability()