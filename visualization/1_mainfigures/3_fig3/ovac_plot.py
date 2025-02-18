import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from util import *
from ase import Atoms
from ase.io import Trajectory, write

import pickle

file_name = '../resources/hist_dict_final.pkl'

# Load the data from the pickle file
with open(file_name, 'rb') as f:
    ovac_data = pickle.load(f)

ovac_structures = Trajectory("../resources/fin_ovac_tot.traj")
original_str=Trajectory('../resources/relaxed_mace_o_deleted.traj')
original_atoms=original_str[0]

all_x=original_atoms.positions[:,0]
all_y=original_atoms.positions[:,1]
all_z=original_atoms.positions[:,2]
cube_xs =[atom.position[0] for atom in original_atoms if atom.position[2]>6]
cube_ys =[atom.position[1] for atom in original_atoms if atom.position[2]>6]
print(np.min(cube_xs),np.max(cube_xs))
print(np.min(cube_ys),np.max(cube_ys))


rod_surf=[]
rod_bulk=[]
rod_subsurf=[]
rod_lower_subsurf=[]
rod_bulk_under_cube=[]
cube_surf=[]
cube_bulk=[]
int_surf=[]
int_bulk=[]
span= 3
for i,(o_index, info) in enumerate(ovac_data.items()):
    x=original_atoms[o_index].position[0]
    y=original_atoms[o_index].position[1]
    z=original_atoms[o_index].position[2]
    if z<6:
        if 1<z<3:
            rod_lower_subsurf.append(info['ovacform'])
        elif 3<z<5:
            rod_subsurf.append(info['ovacform'])
        elif np.min(cube_xs)<x<np.max(cube_xs) and np.min(cube_ys)<y<np.max(cube_ys) and 5<z and 5<z: 
            rod_bulk_under_cube.append(info['ovacform'])
        elif 5<z: 
            rod_surf.append(info['ovacform'])
    elif 6<z<17:
        if np.min(cube_xs)+span<x<np.max(cube_xs)-span and np.min(cube_ys)+span<y<np.max(cube_ys)-span:
            int_bulk.append(info['ovacform'])
        else: 
            int_surf.append(info['ovacform'])
    elif 17<z:
        if np.min(cube_xs)+span<x<np.max(cube_xs)-span and np.min(cube_ys)+span<y<np.max(cube_ys)-span:
            cube_bulk.append(info['ovacform'])
        else: 
            cube_surf.append(info['ovacform'])


new_colors=[colors[2],colors[1],colors[0]]#['black','gray',colors[0]]
def plt_hist_list(energy_list_list,label,alphas,colors):

    min_value = min([min(dataset) for dataset in energy_list_list])
    max_value = max([max(dataset) for dataset in energy_list_list])
    bin_range = np.linspace(min_value, max_value, 100) 
    plt.figure(figsize=(5, 4))
    for i, dataset in enumerate(energy_list_list):
        counts, bins = np.histogram(dataset, bins=bin_range)
        bin_centers = 0.5 * (bins[1:] + bins[:-1])
        plt.bar(bin_centers, counts, width=(bins[1] - bins[0]), color=colors[i], alpha=alphas[i], label=label[i], edgecolor=colors[i], linewidth=1.5, zorder=2+i*0.1)
    plt.tight_layout()  # Add tight layout
    plt.xlabel("Oxygen vacancy fomation Energy (eV)", fontproperties=font_properties_label)
    plt.ylabel("Count", fontproperties=font_properties_label)
    plt.xticks(fontproperties=font_properties_tick)
    plt.yticks(fontproperties=font_properties_tick)
    plt.yscale('log')
    plt.tight_layout()
    plt.ylim(0,100)
    plt.xlim(-2.5,2.5)

    plt.savefig(f'./1_pngs/fig3_b.png', dpi=200)
    plt.show()


alpha=0.5
#plt_hist_list(energy_list_list,label,alphas,new_colors)



def plt_violin_list(energy_list_list, labels, colors):
    plt.figure(figsize=(5, 4))

    # Create the violin plot
    parts = plt.violinplot(energy_list_list, showmeans=False, showmedians=True, vert=True)

    # Customizing the appearance of the violins
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i])  # Set the color of each violin
        pc.set_edgecolor('black')    # Set the edge color
        pc.set_alpha(0.7)            # Set the transparency

    # Set properties for the median line
    for partname in ('cbars', 'cmins', 'cmaxes'):
        vp = parts[partname]
        vp.set_edgecolor('black')
        vp.set_linewidth(1.5)

    for i, dataset in enumerate(energy_list_list):
        x = np.random.normal(i + 1, 0.04, size=len(dataset))  # Add jitter to x values (centered at i+1 with slight randomness)
        plt.scatter(x, dataset, color=colors[i], edgecolor='black', zorder=3, alpha=0.6)


    plt.xticks(np.arange(1, len(labels) + 1), labels)  # Set x-ticks to labels
    plt.xlabel("Site Type", fontsize=12)
    plt.ylabel("Oxygen Vacancy Formation Energy (eV)", fontsize=12)
    plt.ylim(-2.5, 2.5)  # Adjust y-limits based on your data
    plt.tight_layout()

    # Save and show the plot
    plt.savefig(f'./1_pngs/violin_plot.png', dpi=200)
    plt.show()

import matplotlib.pyplot as plt
import numpy as np

def plt_horizontal_violin_list(energy_list_list, labels, colors):
    plt.figure(figsize=(5, 4))
    plt.axvline(x=0, color='gray', linestyle='--', linewidth=1,alpha=0.5)

    # Create the horizontal violin plot
    parts = plt.violinplot(energy_list_list, showmeans=False, vert=False)

    # Customizing the appearance of the violins
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i])  # Set the color of each violin
        pc.set_edgecolor('black')    # Set the edge color
        pc.set_alpha(0.7)            # Set the transparency

    # Set properties for the median line
    for partname in ('cbars', 'cmins', 'cmaxes'):
        vp = parts[partname]
        vp.set_edgecolor('black')
        vp.set_linewidth(1.5)

    # Add scatter points with jitter along the y-axis (since violins are now horizontal)
    for i, dataset in enumerate(energy_list_list):
        y = np.random.normal(i + 1, 0.04, size=len(dataset))  # Add jitter to y values (centered at i+1 with slight randomness)
        plt.scatter(dataset, y, color=colors[i],s=10, edgecolor='black', zorder=3, alpha=0.6)

    # Set y-ticks to labels and adjust axis limits\

    plt.yticks(np.arange(1, len(labels) + 1), labels)  # Set y-ticks to labels
    plt.xlabel("Oxygen Vacancy Formation Energy (eV)", fontproperties=font_properties_label)
    plt.xlim(-2, 5)  # Adjust x-limits based on your data
    plt.tight_layout()
    plt.xticks(fontproperties=font_properties_tick)
    plt.yticks(fontproperties=font_properties_tick)
    # Save and show the plot
    plt.savefig(f'./1_pngs/horizontal_violin_plot_with_scatter.png', dpi=200)
    plt.show()



from matplotlib.colors import BoundaryNorm
from mpl_toolkits.mplot3d import Axes3D  # This import registers the 3D projection

def plot_pos_nei_3d(ovac_data, original_atoms):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')  # Create a 3D subplot

    # Extract 'ovacform' values and prepare colors
    values = [info['ovacform'] for o_index, info in ovac_data.items()]
    min_val = -2 #int(np.floor(np.min(values)))  # Minimum integer value
    max_val = 4 #int(np.ceil(np.max(values)))  # Maximum integer value, ensuring inclusive range

    cmap = plt.cm.inferno_r  # Directly use the magma colormap
    bins = np.arange(min_val, max_val + 1)  # Create integer bins from min to max value
    norm = BoundaryNorm(bins, cmap.N, clip=True)  # Setup the normalization with integer bins

    # Plot each point in 3D space
    for i, (o_index, info) in enumerate(ovac_data.items()):
        x = original_atoms[o_index].position[0]
        y = original_atoms[o_index].position[1]
        z = original_atoms[o_index].position[2]
        ax.scatter(x, y, z, s=30, alpha=0.5, color=cmap(norm(values[i])))

    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')

    # Customize tick labels with font properties
    for t in [ax.xaxis, ax.yaxis, ax.zaxis]:
        t.label.set_fontproperties(font_properties_tick)
        for label in t.get_ticklabels():
            label.set_fontproperties(font_properties_tick)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # You have to set the array for the scalar mappable
    cbar = plt.colorbar(sm, ticks=np.linspace(min_val, max_val, len(bins)), pad=0.05, shrink=0.5)  # Adjust the shrink parameter
    cbar.set_label('Oxygen Vacancy Formation Energy (eV)', fontproperties=font_properties_tick)  # Directly setting the font properties
    cbar.ax.set_yticklabels(['{:.1f}'.format(b) for b in bins], fontproperties=font_properties_tick)
    ax.view_init(elev=15, azim=105)  # Adjust 'elev' and 'azim' to change the viewing angle as desired
    ax.set_xlim(0,55)
    ax.set_ylim(0,55)
    ax.set_zlim(0,40)

    ax.invert_yaxis()
    ax.invert_xaxis()


    plt.savefig(f'./1_pngs/ovac_color.png', dpi=200)

    #plt.tight_layout()  # Add tight layout

    plt.show()

# Example call to the function
#plot_pos_nei_3d(ovac_data, original_atoms)

def plot_position(ovac_data, original_atoms,include_ce='on'):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')  # Create a 3D subplot
    ce_indices=[atoms.index for atoms in original_atoms if atoms.symbol=='Ce']
    # Plot each oxygen atom position in 3D space
    name='wo_Ce'

    if include_ce =='on':
        name='Ce'
        for ce_index in ce_indices:
            x = original_atoms[ce_index].position[0]
            y = original_atoms[ce_index].position[1]
            z = original_atoms[ce_index].position[2]
            ax.scatter(x, y, z, s=120, alpha=0.9, color='#dddcc7')  # Red color for O atoms

    for o_index in ovac_data.keys():
        x = original_atoms[o_index].position[0]
        y = original_atoms[o_index].position[1]
        z = original_atoms[o_index].position[2]
        ax.scatter(x, y, z, s=30, alpha=0.9, color='red')  # Red color for O atoms

    # Set axis labels
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')

    # Customize tick labels with font properties (assuming you have defined font_properties_tick)
    for t in [ax.xaxis, ax.yaxis, ax.zaxis]:
        t.label.set_fontproperties(font_properties_tick)
        for label in t.get_ticklabels():
            label.set_fontproperties(font_properties_tick)

    # Set viewing angle and optionally invert axes
    ax.view_init(elev=15, azim=105)  # Adjust 'elev' and 'azim' as needed
    ax.set_xlim(0,55)
    ax.set_ylim(0,55)
    ax.set_zlim(0,40)
    ax.invert_yaxis()  # Invert y-axis if needed
    ax.invert_xaxis()  # Invert x-axis if needed



    plt.savefig(f'./1_pngs/ovac_before_{name}.png', dpi=200)

    # Display the plot
    plt.show()




def plot_pos_nei_2d(ovac_data, original_atoms):
    # Extract data
    x_positions = []
    z_positions = []
    values = []
    for o_index, info in ovac_data.items():
        x_positions.append(original_atoms[o_index].position[0])
        z_positions.append(original_atoms[o_index].position[2])
        values.append(info['ovacform'])
    
    # Create figure with modified gridspec - making both histograms smaller
    fig = plt.figure(figsize=(5, 5))
    gs = plt.GridSpec(4, 4, 
                     height_ratios=[0.5, 2, 2, 0], 
                     width_ratios=[2, 2, 0.5, 0])  # Added extra small rows/columns
    
    # Create main scatter plot with smaller histograms
    ax = fig.add_subplot(gs[1:3, :2])  # Main plot
    ax_histx = fig.add_subplot(gs[0, :2])  # Top histogram
    ax_histy = fig.add_subplot(gs[1:3, 2])  # Right histogram
    
    # Set up colormap
    min_val = -2
    max_val = 4
    cmap = plt.cm.inferno_r
    bins = np.linspace(min_val, max_val, 100)  # Change 50 to the desired number of bins
    norm = BoundaryNorm(bins, cmap.N, clip=True)
    
    # Main scatter plot
    scatter = ax.scatter(x_positions, z_positions, c=values, 
                        cmap=cmap, norm=norm, s=30, alpha=0.5)
    
    # Histograms with matched limits
    energy_threshold = 0
    mask = np.array(values) < energy_threshold
    exothermic_count = np.sum(mask)  # Count exothermic sites
    
    binwidth = 2
    bins_x = np.arange(0, 55 + binwidth, binwidth)
    bins_y = np.arange(0, 55 + binwidth, binwidth)
    
    ax_histx.hist(np.array(x_positions)[mask], bins=bins_x, color='gray', alpha=0.5)
    ax_histy.hist(np.array(z_positions)[mask], bins=bins_y, 
                  orientation='horizontal', color='gray', alpha=0.5)
    
    # Set matching limits
    ax.set_xlim(0, 55)
    ax.set_ylim(0, 40)  # Adjust if needed
    ax_histx.set_xlim(0, 55)
    ax_histy.set_ylim(0, 40)  # Match with main plot
    
    # Add label for exothermic sites
    ax_histx.set_title(f'Exothermic Ovac Sites (n={exothermic_count})', 
                      fontproperties=font_properties_tick, pad=5)
    
    # Remove ticks from histograms
    ax_histx.xaxis.set_tick_params(labelbottom=False)
    ax_histy.yaxis.set_tick_params(labelleft=False)
    
    # Labels
    ax.set_xlabel('X Axis', fontproperties=font_properties_tick)
    ax.set_ylabel('Z Axis', fontproperties=font_properties_tick)

    
    plt.tight_layout()
    plt.savefig('./1_pngs/ovac_color_2d.png', dpi=200, bbox_inches='tight')
    plt.show()




def plot_pos_nei_2d_surface(energy_list_list, original_atoms, labels):
    # Extract data for surface atoms only
    x_positions = []
    z_positions = []
    values = []
    categories = []  # To keep track of which surface type each point belongs to
    
    # Extract data
    x_positions_total = []
    z_positions_total = []
    for o_index, info in ovac_data.items():
        x_positions_total.append(original_atoms[o_index].position[0])
        z_positions_total.append(original_atoms[o_index].position[2])

    # Process each surface type
    for surface_idx, surface_energies in enumerate(energy_list_list):
        for energy in surface_energies:
            # Find corresponding oxygen index
            for o_index, info in ovac_data.items():
                if abs(info['ovacform'] - energy) < 1e-6:  # Compare with small tolerance
                    x_positions.append(original_atoms[o_index].position[0])
                    z_positions.append(original_atoms[o_index].position[2])
                    values.append(energy)
                    categories.append(surface_idx)
                    break
    
    # Create figure with modified gridspec
    fig = plt.figure(figsize=(5, 5))
    gs = plt.GridSpec(4, 4, 
                     height_ratios=[1, 2, 2, 0], 
                     width_ratios=[2, 2, 1, 0])
    
    # Create main scatter plot with smaller histograms
    ax = fig.add_subplot(gs[1:3, :2])
    ax_histx = fig.add_subplot(gs[0, :2])
    ax_histy = fig.add_subplot(gs[1:3, 2])
    
    # Set up colormap
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Different color for each surface type
    
    
    # Set up colormap
    min_val = -2
    max_val = 4
    cmap = plt.cm.inferno_r
    bins = np.linspace(min_val, max_val, 50)  # Change 50 to the desired number of bins
    norm = BoundaryNorm(bins, cmap.N, clip=True)

    scatter2 = ax.scatter(x_positions_total, z_positions_total, c='white', alpha=0.5, s=30, edgecolor='black')    
    # Main scatter plot
    scatter = ax.scatter(x_positions, z_positions, c=values, 
                        cmap=cmap, norm=norm, s=30, alpha=0.5,zorder=10)
    # Main scatter plot
    # Add legend
    ax.legend(fontsize=8)
    
    # Histograms
    energy_threshold = 0
    mask = np.array(values) < energy_threshold
    exothermic_count = np.sum(mask)
    
    binwidth = 0.5  # Reduced from 2 to 0.5 for finer discretization
    bins_x = np.arange(0, 55 + binwidth, binwidth)
    bins_y = np.arange(0, 40 + binwidth, binwidth)  # Changed upper limit to match ylim
    
    ax_histx.hist(np.array(x_positions)[mask], bins=bins_x, color='gray', alpha=0.5)
    ax_histy.hist(np.array(z_positions)[mask], bins=bins_y, 
                  orientation='horizontal', color='gray', alpha=0.5)
    
    # Set matching limits
    ax.set_xlim(0, 55)
    ax.set_ylim(0, 40)
    ax_histx.set_xlim(0, 55)
    ax_histy.set_ylim(0, 40)
    
    # Add label for exothermic sites
    ax_histx.set_title(f'Exothermic Surface Sites (n={exothermic_count})', 
                      fontproperties=font_properties_tick, pad=5)
    
    # Remove ticks from histograms
    ax_histx.xaxis.set_tick_params(labelbottom=False)
    ax_histy.yaxis.set_tick_params(labelleft=False)
    
    # Labels
    ax.set_xlabel('X Axis', fontproperties=font_properties_tick)
    ax.set_ylabel('Z Axis', fontproperties=font_properties_tick)
    
    plt.tight_layout()
    plt.savefig('./1_pngs/ovac_color_2d_surface.png', dpi=200, bbox_inches='tight')
    plt.show()

energy_list_list = [int_surf, rod_surf, cube_surf]
label = ['interface site', 'rod site', 'cube site']
print(len(int_surf),len(rod_surf),len(cube_surf))
exothermic_int_surf=[]
exothermic_rod_surf=[]
exothermic_cube_surf=[]
for i in int_surf:
    if i<0:
        exothermic_int_surf.append(i)
for i in rod_surf:
    if i<0:
        exothermic_rod_surf.append(i)
for i in cube_surf:
    if i<0:
        exothermic_cube_surf.append(i)
print(len(exothermic_int_surf),len(exothermic_rod_surf),len(exothermic_cube_surf))
#plot_position(ovac_data, original_atoms,'on')
#plot_position(ovac_data, original_atoms,'off')

#plt_violin_list(energy_list_list, label, colors)

#plot_pos_nei_2d_surface(energy_list_list, original_atoms, label)

