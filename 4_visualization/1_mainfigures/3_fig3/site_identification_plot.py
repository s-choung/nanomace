import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from util import *
from ase import Atoms
from ase.io import Trajectory, write
import pickle
from matplotlib.colors import BoundaryNorm
from mpl_toolkits.mplot3d import Axes3D  # This import registers the 3D projection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # Import Poly3DCollection here

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


def plot_position(ovac_data, original_atoms,include_ce='on',name='wo_Ce'):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')  # Create a 3D subplot
    ce_indices=[atoms.index for atoms in original_atoms if atoms.symbol=='Ce']
    # Plot each oxygen atom position in 3D space
    if include_ce =='on':
        for ce_index in ce_indices:
            x = original_atoms[ce_index].position[0]
            y = original_atoms[ce_index].position[1]
            z = original_atoms[ce_index].position[2]
            ax.scatter(x, y, z, s=120, alpha=0.7, color='#FFFDD3')  # Red color for O atoms

    for o_index in ovac_data.keys():
        x = original_atoms[o_index].position[0]
        y = original_atoms[o_index].position[1]
        z = original_atoms[o_index].position[2]
        ax.scatter(x, y, z, s=30, alpha=0.7, color='#BD2A2A')  # Red color for O atoms
 
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
    #plt.show()

    # Display the plot


def generate_3d_plot(ovac_data, original_atoms, max_x, max_y, max_z, cube_criteria, rod_criteria, interface_criteria):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')



    # Iterate over the data and classify based on criteria
    for o_index, info in ovac_data.items():
        x = original_atoms[o_index].position[0]
        y = original_atoms[o_index].position[1]
        z = original_atoms[o_index].position[2]

        # Classify and plot based on criteria
        if cube_criteria(x, y, z, max_x, max_y, max_z):
            ax.scatter(x, y, z, s=30, alpha=0.7, color=colors[2])
        elif rod_criteria(x, y, z, max_x, max_y, max_z):
            ax.scatter(x, y, z, s=30, alpha=0.7, color=colors[1])
        elif interface_criteria(x, y, z, max_x, max_y, max_z):
            ax.scatter(x, y, z, s=30, alpha=0.7, color=colors[0])

    # Set axis labels
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')

    # Set viewing angle and optionally invert axes
    ax.view_init(elev=15, azim=105)
    ax.set_xlim(0, max_x)
    ax.set_ylim(0, max_y)
    ax.set_zlim(0, max_z)
    ax.invert_yaxis()
    ax.invert_xaxis()

    plt.savefig('./1_pngs/ovac_3d_plot_colored.png', dpi=200)
    plt.show()

# Example criteria functions
def cube_criteria(x, y, z, max_x, max_y, max_z):
    return z > 17 and (np.min(cube_xs) < x < np.max(cube_xs)) and (np.min(cube_ys) < y < np.max(cube_ys))

def rod_criteria(x, y, z, max_x, max_y, max_z):
    return z < 6 

def interface_criteria(x, y, z, max_x, max_y, max_z):
    return 6 < z < 17 and (np.min(cube_xs) < x < np.max(cube_xs)) and (np.min(cube_ys) < y < np.max(cube_ys))



def generate_3d_plot_filled(ovac_data, original_atoms, max_x, max_y, max_z, cube_criteria, rod_criteria, interface_criteria):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Get boundaries for each region
    cube_points = []
    rod_points = []
    interface_points = []

    for o_index, info in ovac_data.items():
        x = original_atoms[o_index].position[0]
        y = original_atoms[o_index].position[1]
        z = original_atoms[o_index].position[2]

        if cube_criteria(x, y, z, max_x, max_y, max_z):
            cube_points.append([x, y, z])
        elif rod_criteria(x, y, z, max_x, max_y, max_z):
            rod_points.append([x, y, z])
        elif interface_criteria(x, y, z, max_x, max_y, max_z):
            interface_points.append([x, y, z])

    # Convert to numpy arrays
    cube_points = np.array(cube_points) if cube_points else np.array([[0,0,0]])
    rod_points = np.array(rod_points) if rod_points else np.array([[0,0,0]])
    interface_points = np.array(interface_points) if interface_points else np.array([[0,0,0]])

    def create_box_vertices(points):
        if len(points) <= 1:
            return None
        x_min, x_max = np.min(points[:,0]), np.max(points[:,0])
        y_min, y_max = np.min(points[:,1]), np.max(points[:,1])
        z_min, z_max = np.min(points[:,2]), np.max(points[:,2])
        
        vertices = [
            # bottom
            [[x_min, y_min, z_min], [x_max, y_min, z_min], [x_max, y_max, z_min], [x_min, y_max, z_min]],
            # top
            [[x_min, y_min, z_max], [x_max, y_min, z_max], [x_max, y_max, z_max], [x_min, y_max, z_max]],
            # sides
            [[x_min, y_min, z_min], [x_min, y_min, z_max], [x_max, y_min, z_max], [x_max, y_min, z_min]],
            [[x_max, y_min, z_min], [x_max, y_min, z_max], [x_max, y_max, z_max], [x_max, y_max, z_min]],
            [[x_max, y_max, z_min], [x_max, y_max, z_max], [x_min, y_max, z_max], [x_min, y_max, z_min]],
            [[x_min, y_max, z_min], [x_min, y_max, z_max], [x_min, y_min, z_max], [x_min, y_min, z_min]]
        ]
        return vertices

    def plot_filled_region(vertices, color, alpha=0.3):
        if vertices is not None:
            poly3d = Poly3DCollection(vertices, alpha=alpha)
            poly3d.set_facecolor(color)
            ax.add_collection3d(poly3d)

    # Plot each region
    plot_filled_region(create_box_vertices(cube_points), colors[2])
    plot_filled_region(create_box_vertices(rod_points), colors[1])
    plot_filled_region(create_box_vertices(interface_points), colors[0])

    for o_index in ovac_data.keys():
        x = original_atoms[o_index].position[0]
        y = original_atoms[o_index].position[1]
        z = original_atoms[o_index].position[2]
        ax.scatter(x, y, z, s=30, alpha=0.5, color='gray')  # Red color for O atoms
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')

    ax.view_init(elev=15, azim=105)
    ax.set_xlim(0, max_x)
    ax.set_ylim(0, max_y)
    ax.set_zlim(0, max_z)
    ax.invert_yaxis()
    ax.invert_xaxis()

    plt.savefig('./1_pngs/ovac_3d_plot_filled.png', dpi=200)
    plt.show()


# Call the function with your data
#generate_3d_plot(ovac_data, original_atoms, 55, 55, 40, cube_criteria, rod_criteria, interface_criteria)
generate_3d_plot_filled(ovac_data, original_atoms, 55, 55, 40, cube_criteria, rod_criteria, interface_criteria)

#plot_position(ovac_data, original_atoms,include_ce='off',name='wo_Ce')
#plot_position(ovac_data, original_atoms,include_ce='on',name='w_Ce')
