"""
Read output from SWASH and create a video.

Code created
"""
#import netCDF4 as nc4
#import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from parse import parse
#import matplotlib.dates as mdates
#from matplotlib import ticker, cm, colors, colormaps
from datetime import datetime, timedelta, date
#from scipy.signal import welch, csd, detrend
# from IPython.display import display, clear_output
#import time
import cv2

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# Specify paths
genpath = '/Users/christiehegermiller/Projects/sedcolab-swash/'
runfolder = genpath + 'slope_1on20/'
filename = 'grid_output3.tbl'
figfolder = genpath + 'figures/'

# Read the file into a NumPy array
try:
    data = np.loadtxt(runfolder + filename, skiprows=34)
    print("2D Array from file")
    # print(data)
except FileNotFoundError:
    print(f"File '{filename}' not found.")
except ValueError as e:
    print(f"Error reading the file: {e}")

# Define video parameters
output_file = figfolder + 'waterlevels_1on20.mp4'
frame_rate = 5

# Step 1: Find unique time steps in the array
unique_values = np.unique(data[:, 0])

# Create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
frame_width, frame_height = 640, 400
out = cv2.VideoWriter(output_file, fourcc, frame_rate, (frame_width, frame_height))

# Calculate the figure (plot) size to maintain the aspect ratio
figure_aspect_ratio = frame_width / frame_height
figure_width = 10  # Define the width of the figure (customize as needed)
figure_height = figure_width / figure_aspect_ratio

lines = []  # Store Line2D objects representing the lines

# Loop through each unique value
for time_step in unique_values:
    # Find the indices of each unique value
    indices = np.where(data[:, 0] == time_step)[0]

    data_temp = data[indices, :]

    # Clear the previous plot and create a new one
    plt.clf()

    fig, ax = plt.subplots(1, figsize=(figure_width, figure_height))

    # fig.set_size_inches(8, 4)

    # Add a new line to the plot or update an existing line
    # if time_step < len(lines):
    #     lines[time_step].set_data(data[indices, 1], data[indices, 2], label=f'$\eta$')  # Update an existing line
    # else:
    # line, = ax.plot(data[indices, 1], data[indices, 2], label=f'$\eta$')  # Add a new line
    # lines.append(line)

    ax.plot(data_temp[:, 1], data_temp[:, 2], c='steelblue', lw=1.5, label=f'$\eta$')
    ax.plot(data_temp[:, 1], -data_temp[:, 3], c='k', lw=1, label=f'$h$')
    ax.fill_between(data_temp[:, 1], np.ones_like(data_temp[:, 1]) * -1.3, -data_temp[:, 3], facecolor='gainsboro',
                       edgecolor='None')
    # plot glass window
    ft2m = 1. / .3048
    glassx = [115.41 / ft2m, (115.41 + 48) / ft2m, (115.41 + 48) / ft2m, 115.41 / ft2m]
    glassy = [-1.3, -1.3, 0.2, 0.2]
    ax.fill(glassx, glassy, edgecolor='darkgray', facecolor='None', alpha=.5)

    # plot breaking onset
    ind_bk = np.where(data_temp[:, 7] == 1)[0]
    if len(ind_bk)>0:
        ind_bk = ind_bk[0]
        ax.plot([data_temp[ind_bk, 1], data_temp[ind_bk, 1]], [-1.3, 0.4], '--', c='black', lw=1)

    # plot runup extent
    #ind_hrun = np.where(data_temp[:, 8] == 0)[0][0]
    #ax.plot([data_temp[ind_hrun, 1], data_temp[ind_hrun, 1]], [-1.3, 0.2], c='gainsboro', lw=2)

    ax.set_xlabel(r'\textit{along-flume distance} (m)')
    ax.set_ylabel(r'\textit{[-]} (m)')
    ax.set_xlim(data[indices, 1].min(), data[indices, 1].max())
    ax.set_ylim(-1.3, 0.2)
    plt.title(f'$t$ = {time_step} s')
    # plt.legend(bbox_to_anchor=(0, 1), loc='upper right', ncol=1)

    # Save the figure as an image
    plt.savefig('temp_frame.png', dpi=100)
    #plt.show()

    # Read the image and write it to the video
    frame = cv2.imread('temp_frame.png')
    frame = cv2.resize(frame, (frame_width, frame_height))
    out.write(frame)

    print(time_step)
    plt.close()

# Release the VideoWriter
out.release()

print(f"Video saved as {output_file}")

plt.close(fig)