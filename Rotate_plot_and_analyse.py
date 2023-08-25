# -*- coding: utf-8 -*-
"""

Created on Fri Sep  2 14:03:51 2022
@author: Ronja Paschke

#####################################################
Please cite:
R. R. Paschke, S. Mohr, S. Lange, A. Lange, J. Kozuch
In-situ spectroscopic detection of large-scale 
reorientations of transmembrane α-helices during 
viroporin channel opening
DOI: 10.1101/2023.06.22.546036
#####################################################

"""

import numpy as np
import math
import matplotlib.pyplot as plt
from ase import io

##############################################################################
# FUNCTIONS ##################################################################
##############################################################################


#-----------------------------------------------------------------------------
# create a rotation matrix for rotation around one of the coordinate axes
#-----------------------------------------------------------------------------
# type  'x' or 'y' or 'z' as first component to choose which axis to rotate around
def rotate (axis, angle, coordinates):
    angle = math.radians(angle)
    if axis == 'x':
        Rot_Matrix = np.array([[1,0,0],[0, np.cos(angle), -np.sin(angle)],[0, np.sin(angle), np.cos(angle)]])
    elif axis == 'y':
        Rot_Matrix = np.array([[np.cos(angle), 0, np.sin(angle)],[0, 1, 0],[-np.sin(angle), 0, np.cos(angle)]])
    elif axis == 'z':
        Rot_Matrix = np.array([[np.cos(angle), -np.sin(angle), 0],[np.sin(angle), np.cos(angle), 0],[0, 0, 1]])
    
    trans_coord = []
    trans_coord = np.dot(Rot_Matrix, coordinates.transpose())
    trans_coord = trans_coord.transpose()
    
    return(trans_coord)

#-----------------------------------------------------------------------------
# rotate around a vector
#-----------------------------------------------------------------------------
# input: rotation axis as unit vector, angle in degrees, coordinates as np.array
def rotate_around_vector(unit_vector, angle, coordinates):
    n1 = unit_vector[0]
    n2 = unit_vector[1]
    n3 = unit_vector[2]
    angle = math.radians(angle)
    a11 = n1**2 * (1-np.cos(angle)) + np.cos(angle)
    a12 = n1 * n2 * (1-np.cos(angle)) - n3 * np.sin(angle)
    a13 = n1 * n3 * (1-np.cos(angle)) + n2 * np.sin(angle)
    a21 = n2 * n1 * (1-np.cos(angle)) + n3 * np.sin(angle)
    a22 = n2**2 * (1-np.cos(angle)) + np.cos(angle)
    a23 = n2 * n3 * (1-np.cos(angle)) - n1 * np.sin(angle)
    a31 = n3 * n1 * (1-np.cos(angle)) - n2 * np.sin(angle)
    a32 = n3 * n2 * (1-np.cos(angle)) + n1 * np.sin(angle)
    a33 = n3**2 * (1-np.cos(angle)) + np.cos(angle)
    Rot_Matrix = np.array([[a11, a12, a13],[a21, a22, a23],[a31, a32, a33]])
    
    trans_coord = []
    trans_coord = np.dot(Rot_Matrix, coordinates.transpose())
    trans_coord = trans_coord.transpose()
    
    return(trans_coord)
    
#-----------------------------------------------------------------------------
# Read txt file with atom coordinates
#-----------------------------------------------------------------------------
def read_txt(coordinate_file):
    xyz = []    
    with open (coordinate_file, 'r') as file:
        rows = file.readlines()
        for line in rows[1:]:
            x, y, z = (line.split())
            x = float(x)
            y = float(y)
            z = float(z)
            xyz.append([x, y, z])
            
    xyz = np.array(xyz, float)            
    return xyz
#-----------------------------------------------------------------------------
# Read pdb file with atom coordinates
#-----------------------------------------------------------------------------
def read_pdb(coordinate_file):    
    xyz = []
    with open(coordinate_file, 'r') as pdb_file:
        for line in pdb_file:
            if line.startswith("ATOM"):
                # extract x, y, z coordinates for carbon alpha atoms
                x = float(line[30:38].strip())
                y = float(line[38:46].strip())
                z = float(line[46:54].strip())
                xyz.append([x, y, z])
    
    xyz = np.array(xyz, float)    
    return xyz

#-----------------------------------------------------------------------------
# Read file with info on helix axis
#-----------------------------------------------------------------------------
# read vector of helix axis
def read_vector(helix_axis_file):
    Helix_axis = []
    with open (helix_axis_file, 'r') as file:
        rows = file.readlines()
        for line in rows[1:4]:
            Helix_axis.append(float(line))

    Helix_axis = np.array(Helix_axis, float)   
    return Helix_axis

# read center of helix
def read_center(helix_axis_file):
    center = []
    with open (helix_axis_file, 'r') as file:
        rows = file.readlines()
        for line in rows[9:12]:
            center.append(float(line))
    center = np.array(center, float)
    return center

#-----------------------------------------------------------------------------
# Read transition dipole moments from gaussian output file
#-----------------------------------------------------------------------------   
def read_TDM(freq_out_file):
    # define variables
    result = []
    counter = 1
    frequency = 0        
    # Read Transition Dipole Moments and count modes
    with open (freq_out_file, 'r') as file:
        for line in file:
            line = line.strip(' \t\n\r') # remove spaces, tabs, new lines and returns
            if (line[0:17] == "Dipole derivative"):
                items = line.split()
                mode = counter
                x = float(items[5].replace('D', 'E'))
                y = float(items[6].replace('D', 'E'))
                z = float(items[7].replace('D', 'E'))      
                result.append([mode, frequency, x, y, z])
                counter += 1
    
    results = np.array(result) # turn it into an 2D array
 
    # Read Frequencies            
    f = 0          
    with open (freq_out_file, 'r') as file:
        for line in file:
            line = line.strip(' \t\n\r') # remove spaces, tabs, new lines and returns
            if (line[0:11] == "Frequencies"): 
                items = line.split() # split the line into a list of seperate items
                for item in items[2:5]:
                    frequency = float(item)
                    results[f, 1] = frequency
                    f += 1
    return results

#-----------------------------------------------------------------------------
# Remove all transition dipole moments except for amide I and II
#-----------------------------------------------------------------------------
def only_amide_bands(results):
    short = []
    for row in results:
        if 1600 < row[1] < 1700: # amide I region
            short.append(row)
        elif 1483 < row[1] < 1600: # amide II region
            short.append(row)
    short = np.array(short)
    return short


#-----------------------------------------------------------------------------
# calculate gaussian distribution curves with intensities and mu (wavenumbers)
#-----------------------------------------------------------------------------
# x: dots for the plot (choose range and resolution)
# I: intensities (squareroot of transition dipole moments)
# mu: wavenumbers (x-values where gaussian distribution have their max)
# sig: sigma influences width of gaussian (to make it as close to IR spec as possible)
def gaussian(x, I, mu, sig):
    return I/(np.sqrt(2*np.pi*sig))*np.exp(-0.5*(x - mu)**2/sig**2)

#-----------------------------------------------------------------------------
# calculate angle between two vectors
#-----------------------------------------------------------------------------
def angle_between(a, b):
    angle = np.arccos(np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b)))
    angle = math.degrees(angle)
    if angle > 90:
        angle = 180 - angle
    return angle

#-----------------------------------------------------------------------------
# compute helix axis from coordinates
#-----------------------------------------------------------------------------
def compute_helix_axis(coordinates):
    # compute principal axis matrix
    inertia = np.dot(coordinates.transpose(), coordinates)
    e_values, e_vectors = np.linalg.eig(inertia)
    
    # order eigen values (and eigen vectors)
    # axis1 is the principal axis with the biggest eigen value (eval1)
    order = np.argsort(e_values)
    eval3, eval2, eval1 = e_values[order]
    axis3, axis2, axis1 = e_vectors[:, order].transpose()
    # Inertia axis are now ordered
    return axis1

#-----------------------------------------------------------------------------
# plot gaussian difference spectrum
#-----------------------------------------------------------------------------
def plot_gaussian_diff(TDM, og_TDM, mu):
    i=0
    x = np.linspace(1300, 1800, 400)
    sig = 5
    while i < len(TDM): 
        I1 = TDM[i,2]**2
        # mu = wavenumber        
        if i == 0:     
            y_pol        = gaussian(x, I1, mu[i], sig) 
        elif i > 0:
            y_pol        = y_pol + gaussian(x, I1, mu[i], sig)        
        i += 1
    
    j = 0
    while j < len(og_TDM):
        og_I = og_TDM[j,2]**2
        
        if j == 0:      
            og_y_pol       = gaussian(x, og_I, mu[j], sig)
        elif j > 0:
            og_y_pol       = og_y_pol + gaussian(x, og_I, mu[j], sig)       
        j += 1
    
    y = y_pol - og_y_pol
    #plt.plot(x, y)
    #plt.show()
    return plt.plot(x, y)

#-----------------------------------------------------------------------------
# calculate gaussian distribution, calculate difference and return y-values
#-----------------------------------------------------------------------------
def y_gaussian_diff(TDM, og_TDM, mu):
    i=0
    while i < len(TDM): 
        x = np.linspace(1300, 1800, 400)
        I = TDM[i,2]**2
        I_og = og_TDM[i,2]**2
        sig = 5
        #mu = TDM[i,1]
        
        if i == 0:     
            y_pol        = gaussian(x, I, mu[i], sig)
            og_y_pol     = gaussian(x, I_og, mu[i], sig)
        elif i > 0:
            y_pol        = y_pol + gaussian(x, I, mu[i], sig) 
            og_y_pol     = og_y_pol + gaussian(x, I_og, mu[i], sig)
        i += 1
    
    y = y_pol - og_y_pol
    data = np.array([x, y])
    #plt.plot(x, y)
    #plt.show()
    return data.transpose()

#-----------------------------------------------------------------------------
# calculate gaussian distribution, add 2nd helix TDM and return y-values
#-----------------------------------------------------------------------------
def y_gaussian_whole_prot(TDM1, TDM2, mu1, mu2):
    i = 0
    x = np.linspace(1300, 1800, 400)
    sig = 7
    while i < len(TDM1): 
        I1 = TDM1[i,2]**2
        # mu = wavenumber        
        if i == 0:     
            y_pol        = gaussian(x, I1, mu1[i], sig) 
        elif i > 0:
            y_pol        = y_pol + gaussian(x, I1, mu1[i], sig)        
        i += 1
    
    j = 0
    while j < len(TDM2):
        I2 = TDM2[j,2]**2
        
        if j == 0:      
            y2_pol       = gaussian(x, I2, mu2[j], sig)
        elif j > 0:
            y2_pol       = y2_pol + gaussian(x, I2, mu2[j], sig)       
        j += 1
    y = y_pol + y2_pol
    data = np.array([x, y])

    return data.transpose()

#-----------------------------------------------------------------------------
# integrate area under curve with trapezoid rule
#-----------------------------------------------------------------------------

def integrate(x, y):
   area = 0
   for i in range(1, len(x)):
       dx = x[i] - x[i-1]
       area += dx * (y[i-1] + y[i]) / 2

   return area

##############################################################################
# SCRIPT #####################################################################
##############################################################################

#-----------------------------------------------------------------------------
# Define rotation axis and opening angles and rotate helix onto z-axis
#-----------------------------------------------------------------------------

# read coordinates  
coord = read_txt('6mjh_open_coordinates.txt')
# put center of the coordinates into origin of coordinate system
center = read_center('6mjh_open_helix_axes.txt')
coord = coord - center
# read the principal axis (helix axis)
axis = read_vector('6mjh_open_helix_axes.txt')
# calculate cross product of axis and z-axis to get vertical rotation axis 
z_axis = np.array([0, 0, 1])
rot_axis = np.cross(axis, z_axis)
# make it a unit vector
rot_axis = rot_axis/np.linalg.norm(rot_axis)

# calculate angle between z-axis and helix axis
angle1 = angle_between(z_axis, axis)
# rotate helix along the new rotation axis by angle1
new_coord = rotate_around_vector(rot_axis, angle1, coord)

# coordinates of experimental helix (6mjh) are coord
# the coordinates with helix axis parallel to z axis are new_coord

#-----------------------------------------------------------------------------
# align coordinates of DFT calculated helix to z-axis
#-----------------------------------------------------------------------------

# read coordinates of helix (2l0j) after DFT calculations
sim_coord = read_pdb('2l0j_TM_freq.pdb')
# need the coordinates with helix axis parallel to z axis
# find center 
sim_center = np.mean(sim_coord, 0)
# center with geometric center
sim_coord = sim_coord - sim_center
# compute principal axis of helix
sim_axis = compute_helix_axis(sim_coord)
# find rotation axis
sim_rot_axis = np.cross(z_axis, sim_axis)
# make it a unit vector
sim_rot_axis = sim_rot_axis/np.linalg.norm(sim_rot_axis)
# find angle between helix axis and z-axis
sim_angle1 = angle_between(z_axis, sim_axis)
# rotate helix along the new rotation axis by angle1
new_sim_coord = rotate_around_vector(sim_rot_axis, sim_angle1, sim_coord)

#-----------------------------------------------------------------------------
# Select a point to find position of helix axis on z-axis
#-----------------------------------------------------------------------------
# find points of ILE 32 C-beta
P = new_coord[71]
S = new_sim_coord[53]
# put z = 0, to look at a 2d problem
P[2] = 0
S[2] = 0
# calculate angle between the two location vectors of the points
angle2 = angle_between(P, S)
#print(angle2)
# rotate the simulated helix around z axis by calculated angle 
# to get the start paramaters (start orientation) to then simulate opening
start_sim_coord = rotate('z', angle2, new_sim_coord)

#-----------------------------------------------------------------------------
# Align TDM results to z-axis and rotate to starting position
#-----------------------------------------------------------------------------
# read TDM of simulated and frequency calculated helix 
TDM = read_TDM('2l0j_TM_freq.out')
# align with z-axis
TDM[:,2:5] = TDM[:,2:5] - sim_center
TDM[:,2:5] = rotate_around_vector(sim_rot_axis, sim_angle1, TDM[:,2:5])
# rotate to starting position
TDM[:,2:5] = rotate('z', angle2, TDM[:,2:5])
# shorten results to only amide bands
short_TDM = only_amide_bands(TDM)

#-----------------------------------------------------------------------------
# Read TDM of 2nd helix 
#-----------------------------------------------------------------------------
TDM2 = read_TDM('2nd_Helix_freq.out')
short_TDM2 = only_amide_bands(TDM2)

#-----------------------------------------------------------------------------
# Simulate opening and plot spectra and get analysis data
#-----------------------------------------------------------------------------
# use starting orientation of simualted helix TDM
# TDM[0] = mode; TDM[1] = wavenumber; TDM[2,3,4] = x, y, z of transition dipole moment
# apply step-wise angle to rotate around rotation axis from above

maxima = []
areas = []
spectra = []
fig = plt.figure()
angle_range = np.arange(0, 92, 2) #from .. to... in steps of...
spectra = [np.linspace(1300, 1800, 400)]
diff_spectra = [np.linspace(1300, 1800, 400)]
og_angle = 0
rotated_TDM2 = rotate_around_vector(rot_axis, og_angle, short_TDM[:,2:5])

for angle in angle_range:
    rotated_TDM1 = rotate_around_vector(rot_axis, angle, short_TDM[:,2:5]) # rotate around rot axis
    plot_gaussian_diff(rotated_TDM1, rotated_TDM2, short_TDM[:,1])    
    data = y_gaussian_whole_prot(rotated_TDM1, short_TDM2, short_TDM[:,1], short_TDM2[:,1]) # get y
    
    # Integrate amide I area       
    areaA1 = integrate(data[200:400,0], data[200:400,1])
    areaA2 = integrate(data[0:200,0], data[0:200,1])
    ratio = areaA1/areaA2    
    areas.append([angle, areaA1, areaA2, ratio])
    
    data_diff = y_gaussian_diff(rotated_TDM1, rotated_TDM2, short_TDM[:,1]) # get y for difference spectra    
    data_diff = data_diff.transpose()
    diff_spectra.append(data_diff[1,:])

areas = np.array(areas)
diff_spectra = np.array(diff_spectra)
diff_spectra = diff_spectra.transpose()


fig3 = plt.figure()
plt.plot(areas[:,0], areas[:,3])


# save results into txt file 
with open (f'Areas_{angle_range[0]}to{angle_range[len(angle_range)-1]}.txt', 'w') as output:
    output.write('angle \t Area Amide I \t Max Area II \t Ratio AI/II \n')
    np.savetxt(output, areas)
    

with open (f'Difference_spectra_{angle_range[0]}to{angle_range[len(angle_range)-1]}.txt', 'w') as output:
    output.write('angle range:' + str(angle_range) + '\n')
    np.savetxt(output, diff_spectra)






