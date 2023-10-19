#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 12:14:41 2023

@author: cayesoneira
"""

# Clear all variables from the global scope
globals().clear()

import os
import sys
from datetime import datetime, timedelta

print('--------------------------------------------------------------')
print('--------------------------------------------------------------')

try:
    sys.argv[1]
except IndexError:
    print("Running the file for the last day.")

    # Get the current date
    current_date = datetime.now()
    
    # Calculate the last day of the current month
    last_day = datetime(current_date.year, current_date.month, current_date.day) - timedelta(days=1)
    
    # Format the last day in YYMMDD format manually
    year = last_day.year % 100
    month = last_day.month
    day = last_day.day
    
    # Ensure two-digit formatting
    if year < 10:
        year_str = f'0{year}'
    else:
        year_str = str(year)
    
    if month < 10:
        month_str = f'0{month}'
    else:
        month_str = str(month)
    
    if day < 10:
        day_str = f'0{day}'
    else:
        day_str = str(day)
    
    last_day_formatted = f'{year_str}{month_str}{day_str}'
    
    file_path_input = f'../Data_and_Results/DAQ_Data/merged-{last_day_formatted}.txt'
    print(f"--> Reading 'merged-{last_day_formatted}.txt'")
    
    # show_plots = True
    show_plots = False
else:
    print("Running with input given in an external script.")
    file_path_input = sys.argv[1]
    show_plots = False

print('--------------------------------------------------------------')
print('--------------------------------------------------------------')

# -----------------------------------------------------------------------------
# Some declarations -----------------------------------------------------------
# -----------------------------------------------------------------------------

# Limit the number of counts or read the whole file
limit = False
limit_number = 50000

# Decide whether to represent the 3D plot
plot_trajectories = False

# Decide wheter to represent the T_B vs. T_F correlations
diagnosis = True

are_points_on_same_line_tolerance = 7000

crosstalk_bound = 1.75

interstrip_crosstalk_bound = crosstalk_bound
crosstalk_crosstalk_bound = crosstalk_bound
multiplicity_crosstalk_bound = crosstalk_bound

crosstalk_Q_ratio_bound = 0.025

interstrip_Q_ratio_bound = 0.025
interstrip_Q_one_hit = 60
interstrip_Q_max_lower_bound = 7
interstrip_Q_max_upper_bound = 60

interstrip_Q_min_upper_bound = 30

multiplicity_time_threshold = 0.5   # 0.5 ns * 100 mm / ns = 5 cm

strip_speed = 200
time_to_position_speed = strip_speed

L_strip = 300

x_min = 0
x_max = 300
y_min = 0
y_max = 287

plot_3d_scene_plane_x_min = x_min
plot_3d_scene_plane_x_max = x_max
plot_3d_scene_plane_y_min = y_min
plot_3d_scene_plane_y_max = y_max

is_line_through_plane_plane_x_min = x_min
is_line_through_plane_plane_x_max = x_max
is_line_through_plane_plane_y_min = y_min
is_line_through_plane_plane_y_max = y_max

right_lim_T = 10
left_lim_T = -200

left_lim_Q = -50
right_lim_Q = 400

calibrate_strip_T_percentile = 10
calibrate_strip_T_translation = (x_max-x_min) / strip_speed # Right now is 1.5

calibrate_strip_T_percentile = 2

calibrate_strip_Q_percentile = 5

x_axes_limit_plots = 150

# Initilialize the count of outputs to store them in order
output_order = 1

# -----------------------------------------------------------------------------
# Header-----------------------------------------------------------------------
# -----------------------------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3D
from scipy.ndimage import gaussian_filter
from PyPDF2 import PdfMerger
import time
import shutil
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Image, Table, TableStyle
from reportlab.lib import colors
from scipy.constants import speed_of_light

# To track the duration of this analysis
init_exec_time = time.time()

# Replace 'your_file.dat' with the path to your .dat file
file_path = file_path_input

# Read the .dat file using pandas.read_csv with space delimiter
data = pd.read_csv(file_path, delim_whitespace=True, header=None)

# First time of the data
# init_time = np.array(data.iloc[0, :][0:6],dtype=int)
# init_time = init_time.astype(str).tolist()
# init_time = datetime.strptime(" ".join(init_time[:6]), "%Y %m %d %H %M %S")

# Assuming 'data' is your DataFrame
year, month, day, hour, minute, second = data.iloc[0, :6].astype(int)

# Construct a datetime string in the correct format
datetime_str = f"{year:04d}-{month:02d}-{day:02d} {hour:02d}:{minute:02d}:{second:02d}"

# Parse the string into a datetime object
init_time = pd.to_datetime(datetime_str, format="%Y-%m-%d %H:%M:%S")


first_datetime = init_time

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Some useful functions -------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Charge summary --------------------------------------------------------------
# -----------------------------------------------------------------------------

def charge_summary(data):
    data = data[data > 0]
    data = data[pd.notna(data)]
    
    summary = {
        'Minimum': np.min(data),
        '0.05 quantile': np.percentile(data, 5),
        'Median': np.median(data),
        'Mean': np.mean(data),
        '0.95 quantile': np.percentile(data, 95),
        'Maximum': np.max(data),
        'Standard Deviation': np.std(data)
    }
    return summary

# -----------------------------------------------------------------------------
# Charge summary --------------------------------------------------------------
# -----------------------------------------------------------------------------

def strip_monitoring_summary(data):
    data = data[data != 0]
    data = data[pd.notna(data)]
    data = data[-300 < data]
    data = data[data < 200]
    
    summary = {
        'Minimum': np.min(data),
        '0.05 quantile': np.percentile(data, 5),
        'Median': np.median(data),
        'Mean': np.mean(data),
        '0.95 quantile': np.percentile(data, 95),
        'Maximum': np.max(data),
        'Standard Deviation': np.std(data)
    }
    return summary


# -----------------------------------------------------------------------------
# Progress bar ----------------------------------------------------------------
# -----------------------------------------------------------------------------

def print_bar(progress):
    """
    A progress bar. Take a value of 'progress' between 0 and 100 and print in
    the terminal output a progress bar.

    Parameters
    ----------
    progress : float
        IBetween 0 and 100.

    Returns
    -------
    None.

    """
    bar_width = 70
    pos = int(bar_width * (progress / 100))
    
    sys.stdout.write("[")
    for i in range(bar_width):
        if i < pos:
            sys.stdout.write("=")
        elif i == pos:
            sys.stdout.write(">")
        else:
            sys.stdout.write(" ")
    
    if progress > 100:
        progress = 100
    
    sys.stdout.write(f"] {int(progress)} %\r")
    sys.stdout.flush()


# -----------------------------------------------------------------------------
# Diagonal monitoring  --------------------------------------------------------
# -----------------------------------------------------------------------------

def diagonal_monitoring(x, y, left_lim, right_lim, title):
    
    if left_lim < -100:
        measure_magnitude = "position"
    else:
        measure_magnitude = "charge"
    
    # -------------------------------------------------------------------------
    # 2D ----------------------------------------------------------------------
    # -------------------------------------------------------------------------
    
    # Create a mask for values within a range defined by left_lim and right_lim
    mask = (left_lim < x) & (x < right_lim) & (left_lim < y) & (y < right_lim)
    
    # Apply the mask to both x and y
    x = x[mask]
    y = y[mask]
    
    plt.close()
    plt.figure(figsize=(16,12))
    
    # Create a scatter plot of the filtered data points
    plt.scatter(x, y, s = 1, marker = ',', label='Original Data')

    # Add labels and title
    if left_lim < -100:
        plt.xlabel('Position at Front, F (ns)')
        plt.ylabel('Position at Back, B (ns)')
    else:
        plt.xlabel('Charge at Front, F (ns, AU)')
        plt.ylabel('Charge at Back, B (ns, AU)')
    
    dated_title = f'{title} at {first_datetime}'
    plt.title(dated_title)

    # Add a legend
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    original_string = title
    new_title = '{}_s{}'.format(original_string[0:2], original_string[-1])
    filename = f'{first_datetime}_diagonal_{measure_magnitude}_{new_title}'
    
    # SET THE FORMAT IMAGE TO PNG
    plt.savefig(f'{filename}.png', format="png")
    # Display the plot
    if show_plots: plt.show()
    
    plt.close()
    
    # -------------------------------------------------------------------------
    # Heatmap -----------------------------------------------------------------
    # -------------------------------------------------------------------------
    
    # Create a 2D histogram to calculate point density
    mask = (x != 0) & (y != 0)
    # Apply the mask to both x and y
    x = x[mask]
    y = y[mask]
    
    heatmap, xedges, yedges = np.histogram2d(x, y, bins=200)

    # Apply a logarithmic transformation to the density values
    heatmap = np.log(heatmap + 1)  # Adding 1 to avoid log(0)
    
    plt.close()
    
    # Create a figure and axis for the plot
    fig, ax = plt.subplots(figsize=(18, 12))
    
    # Use Matplotlib's pcolormesh to create a heatmap with log-scaled colors
    cax = ax.pcolormesh(xedges, yedges, heatmap.T, cmap='viridis')
    
    # Add a colorbar to the plot
    plt.colorbar(cax, label='Log Density')
    
    # Set axis labels and title
    ax.set_title('Log-Scaled 2D Point Density Plot')
    
    # Add labels and title
    if left_lim < -100:
        plt.xlabel('Position at Front, F (ns)')
        plt.ylabel('Position at Back, B (ns)')
    else:
        plt.xlabel('Charge at Front, F (ns, AU)')
        plt.ylabel('Charge at Back, B (ns, AU)')
    
    dated_title = f'{title} at {first_datetime}'
    plt.title(dated_title)

    # Add a legend
    plt.grid(True)
    plt.tight_layout()
    
    original_string = title
    new_title = '{}_s{}'.format(original_string[0:2], original_string[-1])
    filename = f'{first_datetime}_{new_title}_diagonal_{measure_magnitude}_histogram'
    
    # SET THE FORMAT IMAGE TO PNG
    plt.savefig(f'{filename}.png', format="png")

    # Display the plot
    if show_plots: plt.show(); plt.close()

    # -------------------------------------------------------------------------
    # Individual histograms ---------------------------------------------------
    # -------------------------------------------------------------------------
    
    # Add labels and title for the histograms
    if left_lim < -100:
        hist_xlabel = 'Position at Front, F (ns)'
        hist_ylabel = 'Position at Back, B (ns)'
    else:
        hist_xlabel = 'Charge at Front, F (ns, AU)'
        hist_ylabel = 'Charge at Back, B (ns, AU)'
    
    # Create histograms for 'x' and 'y' data
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3))
    
    # Some statistics and some labels
    summary_front = strip_monitoring_summary(x)
    summary_back = strip_monitoring_summary(y)
    
    with open(f"stripwise_ascii_monitoring_from_{first_datetime}.txt", 'a') as f:
        print(f"From {first_datetime} \
{new_title} \
{measure_magnitude} \
Front \
{len(x)} \
{summary_front['Minimum']:.5g} \
{summary_front['Median']:.5g} \
{summary_front['Mean']:.5g} \
{summary_front['0.95 quantile']:.5g} \
{summary_front['Maximum']:.5g} \
{summary_front['Standard Deviation']:.5g} \
Back \
{len(y)} \
{summary_back['Minimum']:.5g} \
{summary_back['0.05 quantile']:.5g} \
{summary_back['Median']:.5g} \
{summary_back['Mean']:.5g} \
{summary_back['0.95 quantile']:.5g} \
{summary_back['Maximum']:.5g} \
{summary_back['Standard Deviation']:.5g}", file=f)

    xlabel=f"{len(x)} detections \n\
Median = {summary_front['Median']:.5g} \n\
Mean = {summary_front['Mean']:.5g} \n\
Std. dev. = {summary_front['Standard Deviation']:.5g}"

    ylabel=f"{len(x)} detections \n\
Median = {summary_back['Median']:.5g} \n\
Mean = {summary_back['Mean']:.5g} \n\
Std. dev. = {summary_back['Standard Deviation']:.5g}"
    
    # Plot histograms for 'x' and 'y'
    
    if left_lim < -100:
        # Position
        ax1.hist(x, bins='auto', color='b', alpha=0.7, label = xlabel)
        ax2.hist(y, bins='auto', color='r', alpha=0.7, label = ylabel)
        
        xmin = -160
        xmax = -130
        
        ax1.set_xlim(xmin, xmax)
        ax2.set_xlim(xmin, xmax)
        
    else:
        # Charge
        ax1.hist(x, bins='auto', color='green', alpha=0.7, label = xlabel)
        ax2.hist(y, bins='auto', color='purple', alpha=0.7, label = ylabel)
        
        xmin = 70
        xmax = 200
        
        ax1.set_xlim(xmin, xmax)
        ax2.set_xlim(xmin, xmax)
    
    # Set labels and titles for the histograms
    ax1.set_xlabel(hist_xlabel)
    ax1.set_ylabel('Counts')
    ax1.set_title(f'Front magnitudes, {title} at {first_datetime}')
    legend = ax1.legend(frameon=False)
    # Hide the color boxes in the legend
    for handle in legend.legendHandles:
        handle.set_visible(False)
    
    ax2.set_xlabel(hist_ylabel)
    ax2.set_ylabel('Counts')
    ax2.set_title(f'Back magnitudes, {title} at {first_datetime}')
    # Add a legend
    legend = ax2.legend(frameon=False)
    # Hide the color boxes in the legend
    for handle in legend.legendHandles:
        handle.set_visible(False)
    
    # Save the histograms as images
    plt.savefig(f'{original_string[1:2]}{original_string[-1]}_{filename}_{measure_magnitude}_by_axes.pdf', format="pdf")
    plt.savefig(f'{original_string[1:2]}{original_string[-1]}_{filename}_{measure_magnitude}_by_axes.png', format="png")
    
    # Display the plots
    if show_plots: plt.show(); plt.close()
    
    return


def calculate_charge_between_F_and_B(vector1, vector2):
    """
    Calculate the charge in a strip checking both sides, Front and Back, and
    getting an average value if the difference in the values is small.

    Parameters
    ----------
    vector1 : Array of float64
        DESCRIPTION.
    vector2 : Array of float64
        DESCRIPTION.

    Returns
    -------
    result : Array of float64
        The 'mean' vector.

    """
    result = np.zeros_like(vector1)
    
    for i in range(len(vector1)):
        if vector1[i] == 0:
            result[i] = vector2[i]
        elif vector2[i] == 0:
            result[i] = vector1[i]
        # elif abs(vector1[i] - vector2[i]) / abs(vector1[i]) > 0.1:
        #     result[i] = -2000
        else:
            result[i] = (vector1[i] + vector2[i]) / 2
    return result


def interstrip(Q):
    """
    Check the charge values in order to select if a hit happened in the middle
    of two strips. The conditions are flexible and can be modified in this
    very function.
    
    Parameters
    ----------
    Q : four component float vector
        The charge vector for a certain layer and event.

    Returns
    -------
    bool
        True if the event is interstrip.

    """
    Q_short = Q[Q != 0]
    
    if len(Q_short) != 2:
        # print('Not a two strips event.')
        return False
    
    non_zero_indices_charge = np.nonzero(Q)[0]
    
    # Close in Y: check if the non-zero indices are consecutive
    if abs(non_zero_indices_charge[1] - non_zero_indices_charge[0]) == 1:
        return True
    
    Q_min = np.min(Q_short)
    Q_max = np.max(Q_short)
    Q_total = Q_min + Q_max
    
    crosstalk_bound = interstrip_crosstalk_bound
    Q_ratio_bound = interstrip_Q_ratio_bound
    Q_ratio = Q_min / Q_max
    
    # The mean of charge/event is 30 and it is not usually above 60.
    Q_one_hit = interstrip_Q_one_hit
    
    condition_min = Q_min > crosstalk_bound and Q_min < interstrip_Q_min_upper_bound
    condition_max = Q_max > interstrip_Q_max_lower_bound and Q_max < interstrip_Q_max_upper_bound
    
    if Q_ratio >= Q_ratio_bound and Q_total <= Q_one_hit and condition_max and condition_min:
        return True
    
    return False


def crosstalk(Q):
    
    crosstalk_bound = crosstalk_crosstalk_bound  # Threshold for crosstalk detection
    Q_ratio_bound = crosstalk_Q_ratio_bound  # Threshold for Q ratio
    
    Q_short = Q[Q != 0]
    
    if len(Q_short) != 2:
        # print('Not a two strips event.')
        return
    
    Q_min = np.min(Q_short)
    Q_max = np.max(Q_short)
    Q_ratio = Q_min / Q_max
    
    # 4. Check if the event is crosstalk.
    if Q_min <= crosstalk_bound or Q_ratio <= Q_ratio_bound:
        return True
    
    return False
    

# WORK IN PROGRESS ------ -----------------------------------------------------

def multiplicity(T, Q):
    
    if np.any(Q) < 0:
        return -1
    
    time_threshold = multiplicity_time_threshold
    
    time_count = np.count_nonzero(T)
    charge_count = np.count_nonzero(Q)
    
    non_zero_indices_time = np.nonzero(T)[0]
    non_zero_indices_charge = np.nonzero(Q)[0]
    
    # 1. Check if the same strips are triggered in time and charge
    if not np.array_equal(non_zero_indices_time, non_zero_indices_charge):
        # 1.a. Else there was a mistake.
        return -1
    
    # Case 0: no multiplicity
    if time_count == 0 and charge_count == 0:
        return 0
    # Case 1: multiplicity n=1
    if time_count == 1 and charge_count == 1:
        return 1
    # Case 2: multiplicity n = 1 or 2
    if time_count == 2 and charge_count == 2:
        
        # 2. Check if the strips triggered are consecutive. If they are not,
        # then it can be considered multiplicity 2, since they are far in Y.
        if abs(non_zero_indices_time[1] - non_zero_indices_time[0]) == 1 and \
            abs(non_zero_indices_charge[1] - non_zero_indices_charge[0]) == 1:
                
            T_non_zero = T[T != 0]
            time_diff = abs(T_non_zero[1] - T_non_zero[0])
            
            # 3. See if the points are close in X; if they are far along the
            # strip then it is n = 2.
            if time_diff < time_threshold:
                
                if crosstalk(Q):
                    return 1
                    
                # 5. Check if the event is interstrip. Else it can be
                # considered a n = 2 detection.
                if interstrip(Q):
                    return 1 # Interstrip event
        return 2
    
    if np.count_nonzero(T) == 3 and np.count_nonzero(Q) == 3:
        
        T_non_zero = T[T != 0]
        
        # 2. Check if the strips triggered are consecutive. If they are not,
        # then the strips triggered were 1-2- -4 or 1- -3-4 (other combination
        # of 3 events has 3 strips consecutively triggered).
        if abs(non_zero_indices_time[1] - non_zero_indices_time[0]) == 1 and \
            abs(non_zero_indices_charge[1] - non_zero_indices_charge[0]) == 1 and \
            abs(non_zero_indices_time[2] - non_zero_indices_time[1]) == 1 and \
            abs(non_zero_indices_charge[2] - non_zero_indices_charge[1]) == 1:
            
            time_diff1 = abs(T_non_zero[1] - T_non_zero[0])
            time_diff2 = abs(T_non_zero[2] - T_non_zero[1])
            
            # 3. See if the points are close in X; if two are close and one is
            # far, then we apply the double hit treatment to those two close.
            if time_diff1 > time_threshold and time_diff2 < time_threshold:
                
                try:
                    np.array([T_non_zero[1], T_non_zero[2]])
                except IndexError:
                    T_non_zero = np.array([T_non_zero[1], T_non_zero[2]])
                else:
                    return 2 # An exception that happened only once. Just in case.
                
                Q_short = np.array([Q[1], Q[2]])
                
                if crosstalk(Q_short):
                    return 2 # Crosstalk almost sure.
                    
                # 5. Check if the event is interstrip. Else it can be
                # considered a n = 2 detection.
                if interstrip(Q_short):
                    return 2 # Interstrip event
                return 3
            
            # 3. See if the points are close in X; if two are close and one is
            # far, then we apply the double hit treatment to those two close.
            if time_diff1 < time_threshold and time_diff2 > time_threshold:
                
                T_non_zero = np.array([T_non_zero[0], T_non_zero[1]])
                Q_short = np.array([Q[0], Q[1]])
                
                if crosstalk(Q_short):
                    return 2 # Crosstalk almost sure.
                    
                # 5. Check if the event is interstrip. Else it can be
                # considered a n = 2 detection.
                if interstrip(Q_short):
                    return 2 # Interstrip event
                return 3
            
            
            # 3. See if the points are close in X; if the hits in consecutive
            # strips are far then we consider it is multiplicity n = 3.
            if time_diff1 > time_threshold and time_diff2 > time_threshold:
                return 3
            
            # 3. See if the points are close in X; if the hits in consecutive
            # strips are far then we consider it is multiplicity n = 3.
            if time_diff1 < time_threshold and time_diff2 < time_threshold:
                # THIS INCLUDES A LOT OF CASES, SINCE WE HAVE TO DIFFERENTITATE
                # WHEN HAVING THREE HITS THAT ARE CLOSE IN X AND Y
                
                # We call the stripsnow by A-B-C. Let's see..
                
                # Case 1. A-B-C with the same charge --> no f***** idea what to do
                
                # Case 2: A-B same high charge, C low ---> like a double
                
                # Case 3: B-C same high charge, A low ---> like a double
                
                # Case 4: A-C same high charge, B low ---> this is a multiplicity 2
                
                # Case 5: A high charge, B-C low ---> multiplicity 1
                
                # Case 6: B high charge, A-C low ---> multiplicity 1
                
                # Case 7: C high charge, A-B low ---> multiplicity 1
                
                return 1
        
        # Check if it is the 1-2- -4 case.
        if abs(non_zero_indices_time[1] - non_zero_indices_time[0]) == 1 and \
            abs(non_zero_indices_charge[1] - non_zero_indices_charge[0]) == 1:
            
            T_non_zero = np.array([T_non_zero[0], T_non_zero[1]])
            time_diff = abs(T_non_zero[1] - T_non_zero[0])
            
            # 3. See if the points are close in X; if they are far along the
            # strip then it is n = 2.
            if time_diff < time_threshold: # time_threshold ns * 100 mm / ns = 5 cm
                
                Q_short = np.array([Q[0], Q[1]])
                
                if crosstalk(Q_short):
                    return 2 # Crosstalk almost sure.
                    
                # 5. Check if the event is interstrip. Else it can be
                # considered a n = 2 detection.
                if interstrip(Q_short):
                    return 2 # Interstrip event
            
        # Check if it is the 1- -3-4 case.
        if abs(non_zero_indices_time[2] - non_zero_indices_time[1]) == 1 and \
            abs(non_zero_indices_charge[2] - non_zero_indices_charge[1]) == 1:
            
            try:
                np.array([T_non_zero[1], T_non_zero[2]])
            except IndexError:
                T_non_zero = np.array([T_non_zero[1], T_non_zero[2]])
            else:
                return 2 # An exception that happened only once. Just in case.
            
            time_diff = abs(T_non_zero[1] - T_non_zero[0])
            
            # 3. See if the points are close in X; if they are far along the
            # strip then it is n = 2.
            if time_diff < time_threshold: # time_threshold ns * 100 mm / ns = 5 cm
                
                Q_short = np.array([Q[1], Q[2]])
                
                if crosstalk(Q_short):
                    return 2 # Crosstalk almost sure.
                    
                # 5. Check if the event is interstrip. Else it can be
                # considered a n = 2 detection.
                if interstrip(Q_short):
                    return 2 # Interstrip event
        return 3
    
    if np.count_nonzero(T) == 4 and np.count_nonzero(Q) == 4:
        
        crosstalk_bound = multiplicity_crosstalk_bound  # Threshold for crosstalk detection
        
        T = T[Q < crosstalk_bound]
        Q = Q[Q < crosstalk_bound]
        
        time_count = np.count_nonzero(T)
        charge_count = np.count_nonzero(Q)
        
        non_zero_indices_time = np.nonzero(T)[0]
        non_zero_indices_charge = np.nonzero(Q)[0]
        
        # 1. Check if the same strips are triggered in time and charge
        if not np.array_equal(non_zero_indices_time, non_zero_indices_charge):
            # 1.a. Else there was a mistake.
            return -1
        
        # Case 0: no multiplicity
        if time_count == 0 and charge_count == 0:
            return 0
        # Case 1: multiplicity n=1
        if time_count == 1 and charge_count == 1:
            return 1
        # Case 2: multiplicity n = 1 or 2
        if time_count == 2 and charge_count == 2:
            
            # 2. Check if the strips triggered are consecutive. If they are not,
            # then it can be considered multiplicity 2, since they are far in Y.
            if abs(non_zero_indices_time[1] - non_zero_indices_time[0]) == 1 and \
                abs(non_zero_indices_charge[1] - non_zero_indices_charge[0]) == 1:
                    
                T_non_zero = T[T != 0]
                time_diff = abs(T_non_zero[1] - T_non_zero[0])
                
                # 3. See if the points are close in X; if they are far along the
                # strip then it is n = 2.
                if time_diff < time_threshold: # time_threshold ns * 100 mm / ns = 5 cm
                    
                    if crosstalk(Q):
                        return 1
                        
                    # 5. Check if the event is interstrip. Else it can be
                    # considered a n = 2 detection.
                    if interstrip(Q):
                        return 1 # Interstrip event
            return 2
        
        if np.count_nonzero(T) == 3 and np.count_nonzero(Q) == 3:
            
            T_non_zero = T[T != 0]
            
            # 2. Check if the strips triggered are consecutive. If they are not,
            # then the strips triggered were 1-2- -4 or 1- -3-4 (other combination
            # of 3 events has 3 strips consecutively triggered).
            if abs(non_zero_indices_time[1] - non_zero_indices_time[0]) == 1 and \
                abs(non_zero_indices_charge[1] - non_zero_indices_charge[0]) == 1 and \
                abs(non_zero_indices_time[2] - non_zero_indices_time[1]) == 1 and \
                abs(non_zero_indices_charge[2] - non_zero_indices_charge[1]) == 1:
                
                time_diff1 = abs(T_non_zero[1] - T_non_zero[0])
                time_diff2 = abs(T_non_zero[2] - T_non_zero[1])
                
                # 3. See if the points are close in X; if two are close and one is
                # far, then we apply the double hit treatment to those two close.
                if time_diff1 > time_threshold and time_diff2 < time_threshold:
                    
                    T_non_zero = np.array([T_non_zero[1], T_non_zero[2]])
                    Q_short = np.array([Q[1], Q[2]])
                    
                    if crosstalk(Q_short):
                        return 2 # Crosstalk almost sure.
                        
                    # 5. Check if the event is interstrip. Else it can be
                    # considered a n = 2 detection.
                    if interstrip(Q_short):
                        return 2 # Interstrip event
                    return 3
                
                # 3. See if the points are close in X; if two are close and one is
                # far, then we apply the double hit treatment to those two close.
                if time_diff1 < time_threshold and time_diff2 > time_threshold:
                    
                    T_non_zero = np.array([T_non_zero[0], T_non_zero[1]])
                    Q_short = np.array([Q[0], Q[1]])
                    
                    if crosstalk(Q_short):
                        return 2 # Crosstalk almost sure.
                        
                    # 5. Check if the event is interstrip. Else it can be
                    # considered a n = 2 detection.
                    if interstrip(Q_short):
                        return 2 # Interstrip event
                    return 3
                
                
                # 3. See if the points are close in X; if the hits in consecutive
                # strips are far then we consider it is multiplicity n = 3.
                if time_diff1 > time_threshold and time_diff2 > time_threshold:
                    return 3
                
                # 3. See if the points are close in X; if the hits in consecutive
                # strips are far then we consider it is multiplicity n = 3.
                if time_diff1 < time_threshold and time_diff2 < time_threshold:
                    # THIS INCLUDES A LOT OF CASES, SINCE WE HAVE TO DIFFERENTITATE
                    # WHEN HAVING THREE HITS THAT ARE CLOSE IN X AND Y
                    
                    # We call the stripsnow by A-B-C. Let's see..
                    
                    # Case 1. A-B-C with the same charge --> no f***** idea what to do
                    
                    # Case 2: A-B same high charge, C low ---> like a double
                    
                    # Case 3: B-C same high charge, A low ---> like a double
                    
                    # Case 4: A-C same high charge, B low ---> this is a multiplicity 2
                    
                    # Case 5: A high charge, B-C low ---> multiplicity 1
                    
                    # Case 6: B high charge, A-C low ---> multiplicity 1
                    
                    # Case 7: C high charge, A-B low ---> multiplicity 1
                    
                    return 1
            
            # Check if it is the 1-2- -4 case.
            if abs(non_zero_indices_time[1] - non_zero_indices_time[0]) == 1 and \
                abs(non_zero_indices_charge[1] - non_zero_indices_charge[0]) == 1:
                
                T_non_zero = np.array([T_non_zero[0], T_non_zero[1]])
                time_diff = abs(T_non_zero[1] - T_non_zero[0])
                
                # 3. See if the points are close in X; if they are far along the
                # strip then it is n = 2.
                if time_diff < time_threshold: # time_threshold ns * 100 mm / ns = 5 cm
                    
                    Q_short = np.array([Q[0], Q[1]])
                    
                    if crosstalk(Q_short):
                        return 2 # Crosstalk almost sure.
                        
                    # 5. Check if the event is interstrip. Else it can be
                    # considered a n = 2 detection.
                    if interstrip(Q_short):
                        return 2 # Interstrip event
                
            # Check if it is the 1- -3-4 case.
            if abs(non_zero_indices_time[2] - non_zero_indices_time[1]) == 1 and \
                abs(non_zero_indices_charge[2] - non_zero_indices_charge[1]) == 1:
                
                T_non_zero = np.array([T_non_zero[1], T_non_zero[2]])
                time_diff = abs(T_non_zero[1] - T_non_zero[0])
                
                # 3. See if the points are close in X; if they are far along the
                # strip then it is n = 2.
                if time_diff < time_threshold: # time_threshold ns * 100 mm / ns = 5 cm
                    
                    Q_short = np.array([Q[1], Q[2]])
                    
                    if crosstalk(Q_short):
                        return 2 # Crosstalk almost sure.
                        
                    # 5. Check if the event is interstrip. Else it can be
                    # considered a n = 2 detection.
                    if interstrip(Q_short):
                        return 2 # Interstrip event
            return 3
        
        return 4
    
    return -1

# -----------------------------------------------------------------------------

def time_to_position(T, Q, thick_strip):
    """
    Ultrasimplified version: does not account for interstrip events.
    
    """
    
    r = np.argmax(Q)
    
    # The four strips have a total width of 287 mm
    if thick_strip == 1:
        y_pos = np.array([238, 157.5, 94.5, 31.5])
    elif thick_strip == 4:
        y_pos = np.array([255.5, 192.5, 129.5, 49])
    
    y = y_pos[r]
        
    # X position:
    internal_speed = time_to_position_speed # it's 200 mm/ns
    x_pos = T * internal_speed / 2 # The 1/2 is because the geometric method requires scaling.
    x = x_pos[r]
    
    # Assemble the position vector
    pos = (x, y)
    
    if np.count_nonzero(T) == 0 or np.count_nonzero(Q) == 0:
        pos = (-1000, -1000)
        
    return pos


def are_points_on_same_line(point1, point2, point3):
    """
    Check if three points are approximately on the same line in 3D space.

    Parameters
    ----------
    point1 : list or tuple
        Coordinates of the first point [x, y, z].
    point2 : list or tuple
        Coordinates of the second point [x, y, z].
    point3 : list or tuple
        Coordinates of the third point [x, y, z].

    Returns
    -------
        bool:
            True if the points are on the same line within the tolerance, 
            False otherwise.
    """
    tolerance = are_points_on_same_line_tolerance
    
    vector1 = np.array(point2) - np.array(point1)
    vector2 = np.array(point3) - np.array(point1)

    cross_product = np.cross(vector1, vector2)
    magnitude_cross_product = np.linalg.norm(cross_product)
    
    return magnitude_cross_product < tolerance

# Function to plot the 3D scene
def plot_3d_scene(point1, point2, point3, plane_z):
    """
    Plot in 3D the miniTRASGO, two points and the line joining them.

    Parameters
    ----------
    point1 : Three dimensional array of float64
        First point.
    point2 : Three dimensional array of float64
        Second point.
    point3 : Three dimensional array of float64
        Second point.
    plane_z : float64
        The height of the plane which we want to check if the line joining both
        points passes.

    Returns
    -------
    None.

    """
    plane_x_min = plot_3d_scene_plane_x_min
    plane_x_max = plot_3d_scene_plane_x_max
    plane_y_min = plot_3d_scene_plane_y_min
    plane_y_max = plot_3d_scene_plane_y_max

    # Generate points for the three planes
    x_plane = np.linspace(plane_x_min, plane_x_max, 100)
    y_plane = np.linspace(plane_y_min, plane_y_max, 100)
    x_plane, y_plane = np.meshgrid(x_plane, y_plane)
    z_plane1 = np.ones_like(x_plane) * 0
    z_plane2 = np.ones_like(x_plane) * (100)
    z_plane3 = np.ones_like(x_plane) * (200)
    z_plane4 = np.ones_like(x_plane) * (400)
    
    # Create the 3D plot
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the three planes
    ax.plot_surface(x_plane, y_plane, z_plane1, alpha=0.5, color='blue')
    ax.plot_surface(x_plane, y_plane, z_plane2, alpha=0.5, color='green')
    ax.plot_surface(x_plane, y_plane, z_plane3, alpha=0.5, color='orange')
    ax.plot_surface(x_plane, y_plane, z_plane4, alpha=0.5, color='magenta')
    
    # ------------------------------------------------------------------------------------------------
    # Extend the line in 3D space

    # Calculate direction vector and extension
    extension_factor = 10
    direction_vector = point2 - point1
    extended_point1 = point1 - extension_factor * direction_vector
    extended_point2 = point2 + extension_factor * direction_vector
    
    x1, y1, z1 = point1
    x2, y2, z2 = point2
    
    # Calculate the line equation
    dx = x2 - x1
    dy = y2 - y1
    dz = z2 - z1
    
    # Calculate the parameter t where the line intersects the plane
    t = (plane_z - z1) / dz
    # Calculate the corresponding x and y coordinates
    x_intercept = x1 + t * dx
    y_intercept = y1 + t * dy

    # Plot the original line
    ax.plot([point1[0], point2[0]], [point1[1], point2[1]], [point1[2], point2[2]],
            marker='o', color='red')

    # Plot the extended line
    ax.plot([extended_point1[0], extended_point2[0]], [extended_point1[1], extended_point2[1]],
            [extended_point1[2], extended_point2[2]], marker='o', color='blue', label='Trajectories')

    # Plot the two points
    ax.scatter(point1[0], point1[1], point1[2], color='red' , s=100, label='Point 1')
    ax.scatter(point2[0], point2[1], point2[2], color='blue', s=100, label='Point 2')
    ax.scatter(point3[0], point3[1], point3[2], color='yellow' , s=100, label='Point 3')
    
    ax.scatter(x_intercept, y_intercept, plane_z, color='magenta', s=100,\
               marker='x', label='Line 1-2 intersection at plane')
    
    # ----------------------------------------------------------------------------------------------
    # Extend the line in 3D space

    # Calculate direction vector and extension
    direction_vector = point2 - point3
    extended_point3 = point3 - extension_factor * direction_vector
    extended_point2 = point2 + extension_factor * direction_vector
    
    x1, y1, z1 = point3
    x2, y2, z2 = point2
    
    # Calculate the line equation
    dx = x2 - x1
    dy = y2 - y1
    dz = z2 - z1
    
    # Calculate the parameter t where the line intersects the plane
    t = (plane_z - z1) / dz
    # Calculate the corresponding x and y coordinates
    x_intercept = x1 + t * dx
    y_intercept = y1 + t * dy

    # Plot the original line
    ax.plot([point3[0], point2[0]], [point3[1], point2[1]], [point3[2], point2[2]],
            marker='o', color='red')

    # Plot the extended line
    ax.plot([extended_point3[0], extended_point2[0]], [extended_point3[1], extended_point2[1]],
            [extended_point3[2], extended_point2[2]], marker='o', color='blue')

    # Plot the two points
    ax.scatter(x_intercept, y_intercept, plane_z, color='green', s=100,\
               marker='x', label='Line 2-3 intersection at plane')

    # -----------------------------------------------------------------------------------------------
    # Extend the line in 3D space

    # Calculate direction vector and extension
    direction_vector = point3 - point1
    extended_point1 = point1 - extension_factor * direction_vector
    extended_point3 = point3 + extension_factor * direction_vector
    
    x1, y1, z1 = point1
    x2, y2, z2 = point3
    
    # Calculate the line equation
    dx = x2 - x1
    dy = y2 - y1
    dz = z2 - z1
    
    # Calculate the parameter t where the line intersects the plane
    t = (plane_z - z1) / dz
    # Calculate the corresponding x and y coordinates
    x_intercept = x1 + t * dx
    y_intercept = y1 + t * dy

    # Plot the original line
    ax.plot([point1[0], point3[0]], [point1[1], point3[1]], [point1[2], point3[2]],
            marker='o', color='red')

    # Plot the extended line
    ax.plot([extended_point1[0], extended_point3[0]], [extended_point1[1], extended_point3[1]],
            [extended_point1[2], extended_point3[2]], marker='o', color='blue')

    ax.scatter(x_intercept, y_intercept, plane_z, color='orange', s=100,\
               marker='x', label='Line 1-3 intersection at plane')
    
    # --------------------------------------------------------------------------------------

    # Set labels and legend
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()

    # Set the axes equal
    ax.axis('equal')

    # Set custom limits for each axis
    ax.set_xlim([-5, 305])
    ax.set_ylim([-5, 290])
    ax.set_zlim([5, 400])

    elevation_angle = 200
    azimuth_angle = 140
    ax.view_init(elev=elevation_angle, azim=azimuth_angle)
    
    # Hide grid lines to show the direction of the strips
    ax.xaxis._axinfo['grid']['color'] = (0, 0, 0, 0)
    ax.zaxis._axinfo['grid']['color'] = (0, 0, 0, 0)
    
    ax.set_yticks([0, 75, 150, 225, 300])
    
    # Y-values for the segments
    y_values = [63, 126, 189]
    
    # Iterate through y-values and add segments at both z=100 and z=300
    for y in y_values:
        # Define the endpoints for the segment at z=100
        segment_z100_start = (0, y, 100)
        segment_z100_end = (300, y, 100)
        segment_z100 = Line3D(*zip(segment_z100_start, segment_z100_end), color='blue')
        
        # Define the endpoints for the segment at z=300
        segment_z300_start = (0, y, 400)
        segment_z300_end = (300, y, 400)
        segment_z300 = Line3D(*zip(segment_z300_start, segment_z300_end), color='blue')
        
        # Add the segments to the plot
        ax.add_line(segment_z100)
        ax.add_line(segment_z300)
        
    # Y-values for the segments
    y_values = [98, 161, 224]
    
    # Iterate through y-values and add segments at both z=100 and z=300
    for y in y_values:
        # Define the endpoints for the segment at z=100
        segment_z100_start = (0, y, 0)
        segment_z100_end = (300, y, 0)
        segment_z100 = Line3D(*zip(segment_z100_start, segment_z100_end), color='blue')
        
        # Define the endpoints for the segment at z=300
        segment_z300_start = (0, y, 200)
        segment_z300_end = (300, y, 200)
        segment_z300 = Line3D(*zip(segment_z300_start, segment_z300_end), color='blue')
        
        # Add the segments to the plot
        ax.add_line(segment_z100)
        ax.add_line(segment_z300)
    
    # Show the plot in the end
    if show_plots: plt.show(); plt.close()
    

def is_line_through_plane(point1, point2, plane_z):
    """
    Get two 3D points, check if the line joining them passes through a certain
    bounded rectangle in space.

    Parameters
    ----------
    point1 : Three dimensional array of float64
        First point.
    point2 : Three dimensional array of float64
        Second point.
    plane_z : float64
        The height of the plane which we want to check if the line joining both
        points passes.

    Returns
    -------
    bool
        True if the straight line passes through the plane indicated.

    """
    x1, y1, z1 = point1
    x2, y2, z2 = point2
    
    plane_x_min = is_line_through_plane_plane_x_min
    plane_x_max = is_line_through_plane_plane_x_max
    plane_y_min = is_line_through_plane_plane_y_min
    plane_y_max = is_line_through_plane_plane_y_max

    # Calculate the line equation
    dx = x2 - x1
    dy = y2 - y1
    dz = z2 - z1

    # Calculate the parameter t where the line intersects the plane
    t = (plane_z - z1) / dz

    # Calculate the corresponding x and y coordinates
    x_intercept = x1 + t * dx
    y_intercept = y1 + t * dy

    return (plane_x_min <= x_intercept <= plane_x_max
        and plane_y_min <= y_intercept <= plane_y_max)

# End of the function definition ----------------------------------------------

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Start (data storing) --------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

# My old import snippet
# T4_F_long = np.array(data.iloc[:,7:11],  dtype=np.float64)
# T4_B_long = np.array(data.iloc[:,11:15], dtype=np.float64)
# Q4_F_long = np.array(data.iloc[:,15:19], dtype=np.float64)
# Q4_B_long = np.array(data.iloc[:,19:23], dtype=np.float64)

# T3_F_long = np.array(data.iloc[:,23:27], dtype=np.float64)
# T3_B_long = np.array(data.iloc[:,27:31], dtype=np.float64)
# Q3_F_long = np.array(data.iloc[:,31:35], dtype=np.float64)
# Q3_B_long = np.array(data.iloc[:,35:39], dtype=np.float64)

# T2_F_long = np.array(data.iloc[:,39:43], dtype=np.float64)
# T2_B_long = np.array(data.iloc[:,43:47], dtype=np.float64)
# Q2_F_long = np.array(data.iloc[:,47:51], dtype=np.float64)
# Q2_B_long = np.array(data.iloc[:,51:55], dtype=np.float64)

# T1_F_long = np.array(data.iloc[:,55:59], dtype=np.float64)
# T1_B_long = np.array(data.iloc[:,59:63], dtype=np.float64)
# Q1_F_long = np.array(data.iloc[:,63:67], dtype=np.float64)
# Q1_B_long = np.array(data.iloc[:,67:71], dtype=np.float64)


# -----------------------------------------------------------------------------
# Importing only the TT1 (self-trigger) ---------------------------------------
# -----------------------------------------------------------------------------

mask = data.iloc[:, 6] == 1

# Define the columns for each array
column_ranges = [
    range(7, 11), range(11, 15), range(15, 19), range(19, 23),
    range(23, 27), range(27, 31), range(31, 35), range(35, 39),
    range(39, 43), range(43, 47), range(47, 51), range(51, 55),
    range(55, 59), range(59, 63), range(63, 67), range(67, 71)
]

# Initialize empty lists for each data array
arrays = [np.zeros((0, 4), dtype=np.float64) for _ in range(16)]

# Iterate through the column ranges and append to the lists based on the condition
for i, column_range in enumerate(column_ranges):
    column_data = data[mask].iloc[:, column_range].values.astype(np.float64)
    arrays[i] = column_data

T4_F_ST_long = arrays[0]
T4_B_ST_long = arrays[1]
Q4_F_ST_long = arrays[2]
Q4_B_ST_long = arrays[3]

T3_F_ST_long = arrays[4]
T3_B_ST_long = arrays[5]
Q3_F_ST_long = arrays[6]
Q3_B_ST_long = arrays[7]

T2_F_ST_long = arrays[8]
T2_B_ST_long = arrays[9]
Q2_F_ST_long = arrays[10]
Q2_B_ST_long = arrays[11]

T1_F_ST_long = arrays[12]
T1_B_ST_long = arrays[13]
Q1_F_ST_long = arrays[14]
Q1_B_ST_long = arrays[15]

# -----------------------------------------------------------------------------
# Importing only the TT2 (3 out of four plane coincidence events) -------------
# -----------------------------------------------------------------------------

mask = data.iloc[:, 6] == 1

# Define the columns for each array
column_ranges = [
    range(7, 11), range(11, 15), range(15, 19), range(19, 23),
    range(23, 27), range(27, 31), range(31, 35), range(35, 39),
    range(39, 43), range(43, 47), range(47, 51), range(51, 55),
    range(55, 59), range(59, 63), range(63, 67), range(67, 71)
]

# Initialize empty lists for each data array
arrays = [np.zeros((0, 4), dtype=np.float64) for _ in range(16)]

# Iterate through the column ranges and append to the lists based on the condition
for i, column_range in enumerate(column_ranges):
    column_data = data[mask].iloc[:, column_range].values.astype(np.float64)
    arrays[i] = column_data

T4_F_long = arrays[0]
T4_B_long = arrays[1]
Q4_F_long = arrays[2]
Q4_B_long = arrays[3]

T3_F_long = arrays[4]
T3_B_long = arrays[5]
Q3_F_long = arrays[6]
Q3_B_long = arrays[7]

T2_F_long = arrays[8]
T2_B_long = arrays[9]
Q2_F_long = arrays[10]
Q2_B_long = arrays[11]

T1_F_long = arrays[12]
T1_B_long = arrays[13]
Q1_F_long = arrays[14]
Q1_B_long = arrays[15]

# -----------------------------------------------------------------------------
# Check the data was taken correctly ------------------------------------------
# -----------------------------------------------------------------------------

if diagnosis:
    
    # -------------------------------------------------------------------------
    # Creating the directories for diagnosis ----------------------------------
    # -------------------------------------------------------------------------
    
    # Create a directory name by concatenating the numbers as strings
    directory_name = "Diagonals"

    # Specify the path where you want to create the directory
    base_directory = "../Data_and_Results/"

    # Combine the base directory and the new directory name
    new_directory_path = os.path.join(base_directory, directory_name)

    # Check if the directory already exists
    if not os.path.exists(new_directory_path):
        # Create the directory
        os.mkdir(new_directory_path)
        print(f"Created directory: {new_directory_path}")
    else:
        print(f"Directory already exists: {new_directory_path}")

    # And change to it so the results are stores there
    os.chdir(new_directory_path)

    # -------------------------------------------------------------------------

    # Create a directory name by concatenating the numbers as strings
    directory_name = "Diagonals_tmp"

    # Specify the path where you want to create the directory
    base_directory = os.getcwd()

    # Combine the base directory and the new directory name
    new_directory_path = os.path.join(base_directory, directory_name)

    # Check if the directory already exists
    if not os.path.exists(new_directory_path):
        # Create the directory
        os.mkdir(new_directory_path)
        print(f"Created directory: {new_directory_path}")
    else:
        print(f"Directory already exists: {new_directory_path}")
    
    # And change to it so the results are stored there
    os.chdir(new_directory_path)
    
    print('--------------------------------------------------------------')
    
    with open(f"stripwise_ascii_monitoring_from_{first_datetime}.txt", 'w') as f:
        print("Date \
Strip and RPC \
Position/charge \
Front --> \
Counts \
Minimum \
0.05 quantile \
Median \
Mean \
0.95 quantile \
Maximum \
Std. dev. \
Back --> \
Counts \
Minimum \
0.05 quantile \
Median \
Mean \
0.95 quantile \
Maximum \
Std. dev.", file=f)
    
    diagonal_monitoring(T1_F_long[:,0], T1_B_long[:,0], left_lim_T, right_lim_T, 'T1, strip 1')
    diagonal_monitoring(T1_F_long[:,1], T1_B_long[:,1], left_lim_T, right_lim_T, 'T1, strip 2')
    diagonal_monitoring(T1_F_long[:,2], T1_B_long[:,2], left_lim_T, right_lim_T, 'T1, strip 3')
    diagonal_monitoring(T1_F_long[:,3], T1_B_long[:,3], left_lim_T, right_lim_T, 'T1, strip 4')
    
    diagonal_monitoring(T2_F_long[:,0], T2_B_long[:,0], left_lim_T, right_lim_T, 'T2, strip 1')
    diagonal_monitoring(T2_F_long[:,1], T2_B_long[:,1], left_lim_T, right_lim_T, 'T2, strip 2')
    diagonal_monitoring(T2_F_long[:,2], T2_B_long[:,2], left_lim_T, right_lim_T, 'T2, strip 3')
    diagonal_monitoring(T2_F_long[:,3], T2_B_long[:,3], left_lim_T, right_lim_T, 'T2, strip 4')
    
    diagonal_monitoring(T3_F_long[:,0], T3_B_long[:,0], left_lim_T, right_lim_T, 'T3, strip 1')
    diagonal_monitoring(T3_F_long[:,1], T3_B_long[:,1], left_lim_T, right_lim_T, 'T3, strip 2')
    diagonal_monitoring(T3_F_long[:,2], T3_B_long[:,2], left_lim_T, right_lim_T, 'T3, strip 3')
    diagonal_monitoring(T3_F_long[:,3], T3_B_long[:,3], left_lim_T, right_lim_T, 'T3, strip 4')
    
    diagonal_monitoring(T4_F_long[:,0], T4_B_long[:,0], left_lim_T, right_lim_T, 'T4, strip 1')
    diagonal_monitoring(T4_F_long[:,1], T4_B_long[:,1], left_lim_T, right_lim_T, 'T4, strip 2')
    diagonal_monitoring(T4_F_long[:,2], T4_B_long[:,2], left_lim_T, right_lim_T, 'T4, strip 3')
    diagonal_monitoring(T4_F_long[:,3], T4_B_long[:,3], left_lim_T, right_lim_T, 'T4, strip 4')

    diagonal_monitoring(Q1_F_long[:,0], Q1_B_long[:,0], left_lim_Q, right_lim_Q, 'Q1, strip 1')
    diagonal_monitoring(Q1_F_long[:,1], Q1_B_long[:,1], left_lim_Q, right_lim_Q, 'Q1, strip 2')
    diagonal_monitoring(Q1_F_long[:,2], Q1_B_long[:,2], left_lim_Q, right_lim_Q, 'Q1, strip 3')
    diagonal_monitoring(Q1_F_long[:,3], Q1_B_long[:,3], left_lim_Q, right_lim_Q, 'Q1, strip 4')
    
    diagonal_monitoring(Q2_F_long[:,0], Q2_B_long[:,0], left_lim_Q, right_lim_Q, 'Q2, strip 1')
    diagonal_monitoring(Q2_F_long[:,1], Q2_B_long[:,1], left_lim_Q, right_lim_Q, 'Q2, strip 2')
    diagonal_monitoring(Q2_F_long[:,2], Q2_B_long[:,2], left_lim_Q, right_lim_Q, 'Q2, strip 3')
    diagonal_monitoring(Q2_F_long[:,3], Q2_B_long[:,3], left_lim_Q, right_lim_Q, 'Q2, strip 4')
    
    diagonal_monitoring(Q3_F_long[:,0], Q3_B_long[:,0], left_lim_Q, right_lim_Q, 'Q3, strip 1')
    diagonal_monitoring(Q3_F_long[:,1], Q3_B_long[:,1], left_lim_Q, right_lim_Q, 'Q3, strip 2')
    diagonal_monitoring(Q3_F_long[:,2], Q3_B_long[:,2], left_lim_Q, right_lim_Q, 'Q3, strip 3')
    diagonal_monitoring(Q3_F_long[:,3], Q3_B_long[:,3], left_lim_Q, right_lim_Q, 'Q3, strip 4')
    
    diagonal_monitoring(Q4_F_long[:,0], Q4_B_long[:,0], left_lim_Q, right_lim_Q, 'Q4, strip 1')
    diagonal_monitoring(Q4_F_long[:,1], Q4_B_long[:,1], left_lim_Q, right_lim_Q, 'Q4, strip 2')
    diagonal_monitoring(Q4_F_long[:,2], Q4_B_long[:,2], left_lim_Q, right_lim_Q, 'Q4, strip 3')
    diagonal_monitoring(Q4_F_long[:,3], Q4_B_long[:,3], left_lim_Q, right_lim_Q, 'Q4, strip 4')
    
    # -------------------------------------------------------------------------
    # Monitoring report creation ----------------------------------------------
    # -------------------------------------------------------------------------
    
    # Position ----------------------------------------------------------------
    
    x = [a for a in os.listdir() if a.endswith("position_by_axes.pdf")]
    x = sorted(x, key=lambda pdf: int(pdf.split('_')[0].split('-')[0]))
    
    merger = PdfMerger()

    for pdf in x:
        merger.append(open(pdf, 'rb'))
        
    report_filename=f"report_position_monitoring_at_{first_datetime}.pdf"
    
    with open(report_filename, "wb") as fout:
        merger.write(fout)
        
    position_report = report_filename    
    
    print('----------------------------------------------------------------------')
    print(f"Report stored as '{report_filename}'")
    
    # Joining all the position plots into one pdf
    
    # List of PNG file paths
    z = [a for a in os.listdir() if a.endswith("position_by_axes.png")]
    z = sorted(z, key=lambda png: int(png.split('_')[0].split('-')[0]))
    
    output_filename_position = "combined_1_positions.pdf"
    
    # Define page layout settings
    page_width, page_height = A4
    num_columns = 2
    num_rows = 8
    
    # Calculate the image width and height to maintain the original aspect ratio
    image_width = page_width / num_columns * 0.95
    image_height = page_height / num_rows * 0.95
    
    # Set margin sizes (in inches)
    left_margin = 0.1 * inch
    right_margin = 0.1 * inch
    top_margin = 0.1 * inch
    bottom_margin = 0.1 * inch
    
    # Create a SimpleDocTemplate for the output PDF with custom margins
    doc = SimpleDocTemplate(
        output_filename_position,
        pagesize=(page_width, page_height),
        leftMargin=left_margin,
        rightMargin=right_margin,
        topMargin=top_margin,
        bottomMargin=bottom_margin
    )
    
    # Create a list to hold the Image objects
    elements = []
    
    # Create a 2D list for arranging the images in a grid
    image_grid = [[None for _ in range(num_columns)] for _ in range(num_rows)]
    
    row = 0
    col = 0
    
    for png_file in z:
        img = Image(png_file, width=image_width, height=image_height, kind='proportional')
        image_grid[row][col] = img
        col += 1
        if col >= num_columns:
            col = 0
            row += 1
    
    # Create a Table to arrange the images in a grid
    image_table = Table(image_grid, colWidths=image_width, rowHeights=image_height)
    
    # Apply a TableStyle to remove borders
    style = TableStyle([
        ('GRID', (0, 0), (-1, -1), 0, colors.white),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
    ])
    image_table.setStyle(style)
    
    elements.append(image_table)
    
    # Build the PDF with the images arranged in a grid
    doc.build(elements)
    
    print(f"PNGs combined into {output_filename_position}")


    # Charge ------------------------------------------------------------------
    
    x = [a for a in os.listdir() if a.endswith("charge_by_axes.pdf")]
    
    x = sorted(x, key=lambda pdf: int(pdf.split('_')[0].split('-')[0]))
    
    merger = PdfMerger()

    for pdf in x:
        merger.append(open(pdf, 'rb'))
        
    report_filename=f"report_charge_monitoring_at_{first_datetime}.pdf"
    
    with open(report_filename, "wb") as fout:
        merger.write(fout)

    charge_report = report_filename
    
    print('----------------------------------------------------------------------')
    print(f"Report stored as '{report_filename}'")
    
    # Joining all the charge plots into one pdf
    
    # List of PNG file paths
    z = [a for a in os.listdir() if a.endswith("charge_by_axes.png")]
    z = sorted(z, key=lambda png: int(png.split('_')[0].split('-')[0]))
    
    output_filename_charge = "combined_2_charges.pdf"
    
    # Define page layout settings
    page_width, page_height = A4
    num_columns = 2
    num_rows = 8
    
    # Calculate the image width and height to maintain the original aspect ratio
    image_width = page_width / num_columns * 0.95
    image_height = page_height / num_rows * 0.95
    
    # Set margin sizes (in inches)
    left_margin = 0.1 * inch
    right_margin = 0.1 * inch
    top_margin = 0.1 * inch
    bottom_margin = 0.1 * inch
    
    # Create a SimpleDocTemplate for the output PDF with custom margins
    doc = SimpleDocTemplate(
        output_filename_charge,
        pagesize=(page_width, page_height),
        leftMargin=left_margin,
        rightMargin=right_margin,
        topMargin=top_margin,
        bottomMargin=bottom_margin
    )
    
    # Create a list to hold the Image objects
    elements = []
    
    # Create a 2D list for arranging the images in a grid
    image_grid = [[None for _ in range(num_columns)] for _ in range(num_rows)]
    
    row = 0
    col = 0
    
    for png_file in z:
        img = Image(png_file, width=image_width, height=image_height, kind='proportional')
        image_grid[row][col] = img
        col += 1
        if col >= num_columns:
            col = 0
            row += 1
    
    # Create a Table to arrange the images in a grid
    image_table = Table(image_grid, colWidths=image_width, rowHeights=image_height)
    
    # Apply a TableStyle to remove borders
    style = TableStyle([
        ('GRID', (0, 0), (-1, -1), 0, colors.white),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
    ])
    image_table.setStyle(style)
    
    elements.append(image_table)
    
    # Build the PDF with the images arranged in a grid
    doc.build(elements)
    
    print(f"PNGs combined into {output_filename_charge}")
    
    # Join both position and charge pdfs --------------------------------------
    
    x = [a for a in os.listdir() if a.startswith("combined")]
    
    merger = PdfMerger()

    for pdf in x:
        merger.append(open(pdf, 'rb'))
        
    report_filename=f"report_stripwise_monitoring_at_{first_datetime}.pdf"
    
    with open(report_filename, "wb") as fout:
        merger.write(fout)

    print('----------------------------------------------------------------------')
    print(f"Report stored as '{report_filename}'")
    
    os.remove(position_report)
    os.remove(charge_report)
    
    os.remove(output_filename_position)
    os.remove(output_filename_charge)
    
    # -------------------------------------------------------------------------
    
    os.chdir("..")
    
    file_path = f"Diagonals_from_{first_datetime}"  # Replace with the path to your file

    if os.path.exists(file_path):
        shutil.rmtree(file_path)
        print(f"Directory already exists: '{file_path}'")
    else:
        print(f"Created directory: '{file_path}'")
    
    shutil.copytree("Diagonals_tmp", f"Diagonals_from_{first_datetime}")
    shutil.rmtree("Diagonals_tmp")
    
    os.chdir("..")
    
    current_directory = os.getcwd()
    print(f'Current directory after the diagonals analysis is {current_directory}')
    print('--------------------------------------------------------------')


# -----------------------------------------------------------------------------    
# -----------------------------------------------------------------------------
# Calibration (stripwise) -----------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Position calibration  (stripwise) -------------------------------------------
# -----------------------------------------------------------------------------

def calibrate_strip_T(column_F, column_B):
    """
    Take two vectors of times, the front and back for a certain strip, and
    calibrate the values.

    Parameters
    ----------
    column_B : Array of float64
        The back time vector for a strip.
    column_F : Array of float64
        The back time vector for a strip.

    Returns
    -------
    offset : float64
        The offset to calibrate the times in a certain strip.

    """
    # q is the percentile of the distrib
    q = calibrate_strip_T_percentile
    
    mask = (column_F < 0) & (column_F > -150) & \
           (column_B < 0) & (column_B > -150)
    
    column_B = column_B[mask]
    column_F = column_F[mask]
    
    column = column_F - column_B
    column = np.array(column)
    
    column = column[(np.percentile(column, q) < column) & (column < np.percentile(column, 100 - q))]
    column = column[(np.percentile(column, q) < column) & (column < np.percentile(column, 100 - q))]
    column = column[(np.percentile(column, q) < column) & (column < np.percentile(column, 100 - q))]
    
    offset = np.mean([np.min(column), np.max(column)])
    offset = offset - calibrate_strip_T_translation
    return offset


# We calculate stripwise the calibration parameter, then store it.
calibration_T = []
calibration_t_component = [
    calibrate_strip_T(T1_F_long[:,0], T1_B_long[:,0]),
    calibrate_strip_T(T1_F_long[:,1], T1_B_long[:,1]),
    calibrate_strip_T(T1_F_long[:,2], T1_B_long[:,2]),
    calibrate_strip_T(T1_F_long[:,3], T1_B_long[:,3])
]
calibration_T.append(calibration_t_component)

calibration_t_component = [
    calibrate_strip_T(T2_F_long[:,0], T2_B_long[:,0]),
    calibrate_strip_T(T2_F_long[:,1], T2_B_long[:,1]),
    calibrate_strip_T(T2_F_long[:,2], T2_B_long[:,2]),
    calibrate_strip_T(T2_F_long[:,3], T2_B_long[:,3])
]
calibration_T.append(calibration_t_component)

calibration_t_component = [
    calibrate_strip_T(T3_F_long[:,0], T3_B_long[:,0]),
    calibrate_strip_T(T3_F_long[:,1], T3_B_long[:,1]),
    calibrate_strip_T(T3_F_long[:,2], T3_B_long[:,2]),
    calibrate_strip_T(T3_F_long[:,3], T3_B_long[:,3])
]
calibration_T.append(calibration_t_component)

calibration_t_component = [
    calibrate_strip_T(T4_F_long[:,0], T4_B_long[:,0]),
    calibrate_strip_T(T4_F_long[:,1], T4_B_long[:,1]),
    calibrate_strip_T(T4_F_long[:,2], T4_B_long[:,2]),
    calibrate_strip_T(T4_F_long[:,3], T4_B_long[:,3])
]
calibration_T.append(calibration_t_component)

# Convert the list to a NumPy array
calibration_T = np.array(calibration_T)

# -----------------------------------------------------------------------------
# Charge calibration (stripwise) ----------------------------------------------
# -----------------------------------------------------------------------------

def calibrate_strip_Q(column_B, column_F):
    """
    Calibrate the charge vectors taking the charge values in the front and back
    of a certain strip and return the offset value.

    Parameters
    ----------
    column_B : Array of float64
        The back charge vector for a strip.
    column_F : Array of float64
        The back charge vector for a strip.

    Returns
    -------
    offset : float64
        The offset to calibrate the charge in a certain strip.

    """
    # q is the percentile of the distrib
    q = calibrate_strip_Q_percentile
    
    mask = (column_F != 0) & (column_B != 0)
    
    column_B = column_B[mask]
    column_F = column_F[mask]
    
    # Apply a mask to filter some values of charge.
    mask = (column_F > -20) & (column_F < 300) & \
           (column_B > -20) & (column_B < 300)
    
    column_B = column_B[mask]
    column_F = column_F[mask]
    
    column = calculate_charge_between_F_and_B(column_F, column_B)
    column = np.array(column)
    
    column = column[column < np.percentile(column, q)]
    
    offset = np.max(column)
    return offset


calibration_Q = []
calibration_q_component = [
    calibrate_strip_Q(Q1_F_long[:,0], Q1_B_long[:,0]),
    calibrate_strip_Q(Q1_F_long[:,1], Q1_B_long[:,1]),
    calibrate_strip_Q(Q1_F_long[:,2], Q1_B_long[:,2]),
    calibrate_strip_Q(Q1_F_long[:,3], Q1_B_long[:,3])
]
calibration_Q.append(calibration_q_component)

calibration_q_component = [
    calibrate_strip_Q(Q2_F_long[:,0], Q2_B_long[:,0]),
    calibrate_strip_Q(Q2_F_long[:,1], Q2_B_long[:,1]),
    calibrate_strip_Q(Q2_F_long[:,2], Q2_B_long[:,2]),
    calibrate_strip_Q(Q2_F_long[:,3], Q2_B_long[:,3])
]
calibration_Q.append(calibration_q_component)

calibration_q_component = [
    calibrate_strip_Q(Q3_F_long[:,0], Q3_B_long[:,0]),
    calibrate_strip_Q(Q3_F_long[:,1], Q3_B_long[:,1]),
    calibrate_strip_Q(Q3_F_long[:,2], Q3_B_long[:,2]),
    calibrate_strip_Q(Q3_F_long[:,3], Q3_B_long[:,3])
]
calibration_Q.append(calibration_q_component)

calibration_q_component = [
    calibrate_strip_Q(Q4_F_long[:,0], Q4_B_long[:,0]),
    calibrate_strip_Q(Q4_F_long[:,1], Q4_B_long[:,1]),
    calibrate_strip_Q(Q4_F_long[:,2], Q4_B_long[:,2]),
    calibrate_strip_Q(Q4_F_long[:,3], Q4_B_long[:,3])
]
calibration_Q.append(calibration_q_component)

# Convert the list to a NumPy array
calibration_Q = np.array(calibration_Q)


# -----------------------------------------------------------------------------
# Incident time calibration ---------------------------------------------------
# -----------------------------------------------------------------------------
# Ti1 = (T1_F_long + T1_B_long - L_strip / strip_speed) / 2
# Ti2 = (T2_F_long + T2_B_long - L_strip / strip_speed) / 2
# Ti3 = (T3_F_long + T3_B_long - L_strip / strip_speed) / 2
# Ti4 = (T4_F_long + T4_B_long - L_strip / strip_speed) / 2

# T1_cal_0 = T1_F_long - T1_B_long - calibration_T[0,:]
# T1_cal_0 = T1_F_long - T1_B_long - calibration_T[0,:]
# T1_cal_0 = T1_F_long - T1_B_long - calibration_T[0,:]
# T1_cal_0 = T1_F_long - T1_B_long - calibration_T[0,:]

# Ti1[(Ti1 > -80) | (Ti1 < -160)] = 0
# Ti2[(Ti2 > -80) | (Ti2 < -160)] = 0
# Ti3[(Ti3 > -80) | (Ti3 < -160)] = 0
# Ti4[(Ti4 > -80) | (Ti4 < -160)] = 0

# v = T1_cal_0
# w = T1_F_long - T1_B_long
# strip = 2

# # v[(v > -80) | (v < -160)] = 0

# v[(v > -100) | (v < 100)] = 0
# v = v[:,strip-1][v[:,strip-1] != 0]

# w[(w > -80) | (w < -160)] = 0
# w = w[:,strip-1][w[:,strip-1] != 0]

# # v = strip_speed * v/2

# plt.close()
# fig = plt.figure(figsize=(10,7))
# plt.hist(v, bins='auto', color='red', alpha=0.7)
# # plt.hist(w, bins='auto', color='b', alpha=0.7)
# plt.xlabel('Charge (AU)')
# plt.ylabel('Counts')
# plt.title('...')
# plt.tight_layout()
# if show_plots: plt.show(); plt.close()

# # for i in range(len(Ti1)):
    
# Ti1_row = Ti1[i,:]
# Ti2_row = Ti2[i,:]
# Ti3_row = Ti3[i,:]
# Ti4_row = Ti4[i,:]

# T = [Ti1_row, Ti2_row, Ti3_row, Ti4_row]

# eps = np.zeros((4,4))

# # Define the known values
# speed_of_light_mm_ns = speed_of_light / 1e6
# z_values = [0, 100, 200, 400]  # Corresponding z values for Ti1, Ti2, Ti3, Ti4

# def incident(i,j):
    
#     delta_z = abs(z_values[i] - z_values[j])
#     eps = delta_z / speed_of_light + abs(T[i] - T[j])
    
#     eps 
#     return eps

# combinations = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]

# # Fill the matrix with values
# for i, j in combinations:
#     eps = incident(i,j)

    

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Define some variables that will be needed -----------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

raw_events = 0
semi_filtered_events = 0
filtered_events = 0
strongly_filtered_events = 0

# We inilitialize to 1 since we want to avoid division by 0
total_crossing_muons_1 = 1
total_crossing_muons_2 = 1
total_crossing_muons_3 = 1
total_crossing_muons_4 = 1

detected_1 = 0
detected_2 = 0
detected_3 = 0
detected_4 = 0

total_crossing_muons_1_naive = 1
total_crossing_muons_2_naive = 1
total_crossing_muons_3_naive = 1
total_crossing_muons_4_naive = 1

detected_1_naive = 0
detected_2_naive = 0
detected_3_naive = 0
detected_4_naive = 0

total_muons_1 = 1
total_muons_2 = 1
total_muons_3 = 1
total_muons_4 = 1

strip_time_fail = 0
strip_charge_fail = 0

position_not_asigned = 0
fully_tapped_events = 0
half_tapped_events = 0


# Magnitudes for periodic (minutal, hourly, etc) calculations -----------------

# Hourly
hourly_total_crossing_muons_1 = 1
hourly_total_crossing_muons_2 = 1
hourly_total_crossing_muons_3 = 1
hourly_total_crossing_muons_4 = 1

hourly_detected_1 = 0
hourly_detected_2 = 0
hourly_detected_3 = 0
hourly_detected_4 = 0

total_number_of_events = data.shape[0]
total_TT1 = 0
total_TT2 = 0

Q1_hourly = np.zeros(4)
Q2_hourly = np.zeros(4)
Q3_hourly = np.zeros(4)
Q4_hourly = np.zeros(4)

# Quarters
multiplicities_quarters = np.zeros(4)

# -----------------------------------------------------------------------------

T1_long = np.zeros(4)
T2_long = np.zeros(4)
T3_long = np.zeros(4)
T4_long = np.zeros(4)

T1_cal_long = np.zeros(4)
T2_cal_long = np.zeros(4)
T3_cal_long = np.zeros(4)
T4_cal_long = np.zeros(4)

T1_F_filtered_long = np.zeros(4)
T1_B_filtered_long = np.zeros(4)
T2_F_filtered_long = np.zeros(4)
T2_B_filtered_long = np.zeros(4)
T3_F_filtered_long = np.zeros(4)
T3_B_filtered_long = np.zeros(4)
T4_F_filtered_long = np.zeros(4)
T4_B_filtered_long = np.zeros(4)

Q1_cal_long = np.zeros(4)
Q2_cal_long = np.zeros(4)
Q3_cal_long = np.zeros(4)
Q4_cal_long = np.zeros(4)

T1_ST_long = np.zeros(4)
T2_ST_long = np.zeros(4)
T3_ST_long = np.zeros(4)
T4_ST_long = np.zeros(4)

T1_cal_ST_long = np.zeros(4)
T2_cal_ST_long = np.zeros(4)
T3_cal_ST_long = np.zeros(4)
T4_cal_ST_long = np.zeros(4)

Q1_cal_ST_long = np.zeros(4)
Q2_cal_ST_long = np.zeros(4)
Q3_cal_ST_long = np.zeros(4)
Q4_cal_ST_long = np.zeros(4)

Q1_ind_strip = np.zeros(4)
Q2_ind_strip = np.zeros(4)
Q3_ind_strip = np.zeros(4)
Q4_ind_strip = np.zeros(4)

Q1_double_strip = np.zeros(4)
Q2_double_strip = np.zeros(4)
Q3_double_strip = np.zeros(4)
Q4_double_strip = np.zeros(4)

Q1_triple_strip = np.zeros(4)
Q2_triple_strip = np.zeros(4)
Q3_triple_strip = np.zeros(4)
Q4_triple_strip = np.zeros(4)

Q1_quad_strip = np.zeros(4)
Q2_quad_strip = np.zeros(4)
Q3_quad_strip = np.zeros(4)
Q4_quad_strip = np.zeros(4)

Q1_n0_strip = np.zeros(4)
Q2_n0_strip = np.zeros(4)
Q3_n0_strip = np.zeros(4)
Q4_n0_strip = np.zeros(4)

Q1_n1_strip = np.zeros(4)
Q2_n1_strip = np.zeros(4)
Q3_n1_strip = np.zeros(4)
Q4_n1_strip = np.zeros(4)

Q1_n2_strip = np.zeros(4)
Q2_n2_strip = np.zeros(4)
Q3_n2_strip = np.zeros(4)
Q4_n2_strip = np.zeros(4)

Q1_n3_strip = np.zeros(4)
Q2_n3_strip = np.zeros(4)
Q3_n3_strip = np.zeros(4)
Q4_n3_strip = np.zeros(4)

Q1_n4_strip = np.zeros(4)
Q2_n4_strip = np.zeros(4)
Q3_n4_strip = np.zeros(4)
Q4_n4_strip = np.zeros(4)

Q1_strip = np.zeros(4)
Q2_strip = np.zeros(4)
Q3_strip = np.zeros(4)
Q4_strip = np.zeros(4)

Q1_interstrip = np.zeros(4)
Q2_interstrip = np.zeros(4)
Q3_interstrip = np.zeros(4)
Q4_interstrip = np.zeros(4)

multiplicities = np.zeros(4)
multi = np.zeros(4)

positions = np.zeros(3)

positions_detected = np.zeros(3)
positions_non_detected = np.zeros(3)

stacked_positions = np.zeros(3)

j = 0

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Creating the directories and files ------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

# Create a directory name by concatenating the numbers as strings
directory_name = "DAQ_data_analysis_results"

# Specify the path where you want to create the directory
os.chdir("../Data_and_Results")
base_directory = os.getcwd()

# Combine the base directory and the new directory name
new_directory_path = os.path.join(base_directory, directory_name)

# Check if the directory already exists
if not os.path.exists(new_directory_path):
    # Create the directory
    os.mkdir(new_directory_path)
    print(f"Created directory: {new_directory_path}")
else:
    print(f"Directory already exists: {new_directory_path}")

# And change directory to it so the results are stored in there
os.chdir(new_directory_path)

# -----------------------------------------------------------------------------

# Create a directory name by concatenating the numbers as strings
directory_name = "DAQ_data_analysis_results_tmp"

# Specify the path where you want to create the directory
base_directory = os.getcwd()

# Combine the base directory and the new directory name
new_directory_path = os.path.join(base_directory, directory_name)

# Check if the directory already exists
if not os.path.exists(new_directory_path):
    # Create the directory
    os.mkdir(new_directory_path)
    print(f"Created directory: {new_directory_path}")
else:
    print(f"Directory already exists: {new_directory_path}")

# And change to it so the results are stored there
os.chdir(new_directory_path)

# -----------------------------------------------------------------------------
# Some files to store data ----------------------------------------------------
# -----------------------------------------------------------------------------

# Rates file
with open("rates_tmp.txt", 'w') as f:
    print(f"Rates from {init_time}; Rates filtered, Rates raw, Rates strongly filtered", file=f)
    
# Efficiencies file
with open("efficiencies_tmp.txt", 'w') as f:
    print(f"Efficiencies from {init_time}; T1, T2, T3, T4", file=f)

# Charges file
with open("charges_T1_tmp.txt", 'w') as f:
    print(f"Charges in T1 from {init_time}; Minimum 0.05 quantile, Median, Mean, 0.95 quantile, Maximum, Standard Deviation", file=f)
    
with open("charges_T2_tmp.txt", 'w') as f:
    print(f"Charges in T2 from {init_time}; Minimum 0.05 quantile, Median, Mean, 0.95 quantile, Maximum, Standard Deviation", file=f)
    
with open("charges_T3_tmp.txt", 'w') as f:
    print(f"Charges in T3 from {init_time}; Minimum 0.05 quantile, Median, Mean, 0.95 quantile, Maximum, Standard Deviation", file=f)
    
with open("charges_T4_tmp.txt", 'w') as f:
    print(f"Charges in T4 from {init_time}; Minimum 0.05 quantile, Median, Mean, 0.95 quantile, Maximum, Standard Deviation", file=f)

# Multiplicities file
with open("mean_multiplicities_tmp.txt", 'a') as f:
    print(f"Mean multiplicities from {init_time}; \
Mean multiplicity in all layers, T1, T2, T3, T4; \
Removing zeroes in all layers, T1, T2, T3, T4;", file=f)


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# We start the proper analysis ------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

init_time_hourly = init_time
init_time_quarters = init_time

new_ratio = 0

print('--------------------------------------------------------------')
print(f'{total_number_of_events} events in the file...')
print('--------------------------------------------------------------')

# Row by row, i.e. event by event. This means the calibration has to be done
# previously, but once done the data treatment is more intuitive event-wise.
for i in range(total_number_of_events):
    
    # Just a progress bar
    if round(i/total_number_of_events * 100) > new_ratio:
        print('------------------------------------------------------------------------')
        new_ratio = round(i / total_number_of_events * 100)
        print("Event ", i)
        print_bar(new_ratio)
        print('\n')
    
    # Count every iteration. This is used in the raw rate calculation.
    raw_events += 1
    
    # -------------------------------------------------------------------------
    # TT2 STUDY ---------------------------------------------------------------
    # -------------------------------------------------------------------------
    
    if data.iloc[i, :][6] == 2:
        total_TT2 = total_TT2 + 1
        
        # Convert the slices to NumPy arrays with the correct data type
        T1_F = T1_F_long[i, :]
        T1_B = T1_B_long[i, :]
        Q1_F = Q1_F_long[i, :]
        Q1_B = Q1_B_long[i, :]
        
        T2_F = T2_F_long[i, :]
        T2_B = T2_B_long[i, :]
        Q2_F = Q2_F_long[i, :]
        Q2_B = Q2_B_long[i, :]
        
        T3_F = T3_F_long[i, :]
        T3_B = T3_B_long[i, :]
        Q3_F = Q3_F_long[i, :]
        Q3_B = Q3_B_long[i, :]
        
        T4_F = T4_F_long[i, :]
        T4_B = T4_B_long[i, :]
        Q4_F = Q4_F_long[i, :]
        Q4_B = Q4_B_long[i, :]
        
        # ---------------------------------------------------------------------
        # Filtering and applying calibrations ---------------------------------
        # ---------------------------------------------------------------------
        
        # Time ----------------------------------------------------------------
        
        # We filter the values outside the TRB time window
        condition_met = (
            (T1_F > 0).any() or
            (T1_B > 0).any() or
            (T2_F > 0).any() or
            (T2_B > 0).any() or
            (T3_F > 0).any() or
            (T3_B > 0).any() or
            (T4_F > 0).any() or
            (T4_B > 0).any() or
            (T1_F < -200).any() or
            (T1_B < -200).any() or
            (T2_F < -200).any() or
            (T2_B < -200).any() or
            (T3_F < -200).any() or
            (T3_B < -200).any() or
            (T4_F < -200).any() or
            (T4_B < -200).any()
        )
        
        if condition_met:
            strip_time_fail += 1
            continue
        
        T1 = T1_F - T1_B
        T2 = T2_F - T2_B
        T3 = T3_F - T3_B
        T4 = T4_F - T4_B
        
        del T1_F, T1_B, T2_F, T2_B, T3_F, T3_B, T4_F, T4_B
        
        condition_met = (
            (T1 > 4).any() or
            (T2 > 4).any() or
            (T3 > 4).any() or
            (T4 > 4).any() or
            (T1 < -4).any() or
            (T2 < -4).any() or
            (T3 < -4).any() or
            (T4 < -4).any())
        
        if condition_met:
            strip_time_fail += 1
            continue
        
        # Time calibration:
        T1_cal = np.where(T1 != 0, T1 - calibration_T[0,:], 0)
        T2_cal = np.where(T2 != 0, T2 - calibration_T[1,:], 0)
        T3_cal = np.where(T3 != 0, T3 - calibration_T[2,:], 0)
        T4_cal = np.where(T4 != 0, T4 - calibration_T[3,:], 0)
        
        del T1, T2, T3, T4
        
        condition_met = (
            (T1_cal < 0).any() or
            (T2_cal < 0).any() or
            (T3_cal < 0).any() or
            (T4_cal < 0).any() or
            (T1_cal > 3).any() or
            (T2_cal > 3).any() or
            (T3_cal > 3).any() or
            (T4_cal > 3).any())
        
        if condition_met:
            strip_time_fail += 1
            continue
        
        T1_cal_ST_long = np.vstack((T1_cal_ST_long, T1_cal))
        T2_cal_ST_long = np.vstack((T2_cal_ST_long, T2_cal))
        T3_cal_ST_long = np.vstack((T3_cal_ST_long, T3_cal))
        T4_cal_ST_long = np.vstack((T4_cal_ST_long, T4_cal))
        
        # Charge ------------------------------------------------------------------
        
        # Average charge: if one of the values is zero or the values are very
        # different then we take one of them.
        Q1 = calculate_charge_between_F_and_B(Q1_F, Q1_B)
        Q2 = calculate_charge_between_F_and_B(Q2_F, Q2_B)
        Q3 = calculate_charge_between_F_and_B(Q3_F, Q3_B)
        Q4 = calculate_charge_between_F_and_B(Q4_F, Q4_B)
        
        # We filter the values outside the TRB time window
        condition_met = (
            (Q1 > 300).any() or
            (Q2 > 300).any() or
            (Q3 > 300).any() or
            (Q4 > 300).any() or
            (Q1 < 0).any() or
            (Q2 < 0).any() or
            (Q3 < 0).any() or
            (Q4 < 0).any())
        
        if condition_met:
            strip_charge_fail += 1
            continue
        
        # Charge calibration
        Q1_cal = np.where(Q1 != 0, Q1 - calibration_Q[0,:], 0)
        Q2_cal = np.where(Q2 != 0, Q2 - calibration_Q[1,:], 0)
        Q3_cal = np.where(Q3 != 0, Q3 - calibration_Q[2,:], 0)
        Q4_cal = np.where(Q4 != 0, Q4 - calibration_Q[3,:], 0)
        
        del Q1, Q2, Q3, Q4
        
        Q1_cal_ST_long = np.vstack((Q1_cal_ST_long, Q1_cal))
        Q2_cal_ST_long = np.vstack((Q2_cal_ST_long, Q2_cal))
        Q3_cal_ST_long = np.vstack((Q3_cal_ST_long, Q3_cal))
        Q4_cal_ST_long = np.vstack((Q4_cal_ST_long, Q4_cal))
        
        continue
        # Self-trigger study is finished. Back to coincidence events.
    
    
    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    # To check when a certain time passes to store quantities vs. time --------
    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    
    new_time_components = data.iloc[i, :6].astype(int)
    # Construct a datetime string in the correct format
    datetime_str = f"{new_time_components[0]:04d}-{new_time_components[1]:02d}-{new_time_components[2]:02d} {new_time_components[3]:02d}:{new_time_components[4]:02d}:{new_time_components[5]:02d}"
    # Parse the string into a datetime object
    new_time = pd.to_datetime(datetime_str, format="%Y-%m-%d %H:%M:%S")
    
    # -------------------------------------------------------------------------
    # Hourly magnitudes -------------------------------------------------------
    # -------------------------------------------------------------------------
    
    time_difference_hourly = (new_time - init_time_hourly).total_seconds()
    
    # Different time intervals can be considered
    if time_difference_hourly >= 3600:
        
        print('--------------------------------------------------------------')
        print('An hour of events')
        print('--------------------------------------------------------------')
        
        time_interval = time_difference_hourly / 3600
        
        # ---------------------------------------------------------------------
        # Rates ---------------------------------------------------------------
        # ---------------------------------------------------------------------
        
        raw_rate = raw_events / time_interval
        raw_events = 0
        
        filtered_rate = filtered_events / time_interval
        filtered_events = 0
        
        strongly_filtered_rate = strongly_filtered_events / time_interval
        strongly_filtered_events = 0
        
        print('--------------------------------------------------------------')
        print(f'Raw rate is {raw_rate} cts/hr')
        print(f'Filtered rate is {filtered_rate} cts/hr')
        print(f'Strongly filtered rate is {strongly_filtered_rate} cts/hr')
        print('--------------------------------------------------------------')
        
        
        with open("rates_tmp.txt", 'a') as f:
            print(f"From {init_time_hourly} to {new_time} {filtered_rate} {raw_rate} {strongly_filtered_rate}", file=f)
        
        # ---------------------------------------------------------------------
        # Efficiencies --------------------------------------------------------
        # ---------------------------------------------------------------------
        
        hourly_eff1 = hourly_detected_1 / hourly_total_crossing_muons_1
        hourly_eff2 = hourly_detected_2 / hourly_total_crossing_muons_2
        hourly_eff3 = hourly_detected_3 / hourly_total_crossing_muons_3
        hourly_eff4 = hourly_detected_4 / hourly_total_crossing_muons_4
        
        print('Efficiencies are:')
        print(f'{hourly_eff1} {hourly_eff2} {hourly_eff3} {hourly_eff4}')
        print('--------------------------------------------------------------')
        
        with open("efficiencies_tmp.txt", 'a') as f:
            print(f"From {init_time_hourly} to {new_time} {hourly_eff1} {hourly_eff2} {hourly_eff3} {hourly_eff4}", file=f)
        
        hourly_total_crossing_muons_1 = 0
        hourly_total_crossing_muons_2 = 0
        hourly_total_crossing_muons_3 = 0
        hourly_total_crossing_muons_4 = 0

        hourly_detected_1 = 0
        hourly_detected_2 = 0
        hourly_detected_3 = 0
        hourly_detected_4 = 0
        
        # ---------------------------------------------------------------------
        # Charges -------------------------------------------------------------
        # ---------------------------------------------------------------------
        
        Q1_hourly = Q1_hourly[1:,:]
        Q2_hourly = Q2_hourly[1:,:]
        Q3_hourly = Q3_hourly[1:,:]
        Q4_hourly = Q4_hourly[1:,:]
        
        Q1_summed = np.sum(Q1_hourly, axis=1)
        Q2_summed = np.sum(Q2_hourly, axis=1)
        Q3_summed = np.sum(Q3_hourly, axis=1)
        Q4_summed = np.sum(Q4_hourly, axis=1)
        
        summary_1 = charge_summary(Q1_summed)
        summary_2 = charge_summary(Q2_summed)
        summary_3 = charge_summary(Q3_summed)
        summary_4 = charge_summary(Q4_summed)
        
        with open("charges_T1_tmp.txt", 'a') as f:
            print(f"From {init_time_hourly} to {new_time} {summary_1['Minimum']} {summary_1['0.05 quantile']} {summary_1['Median']} {summary_1['Mean']} {summary_1['0.95 quantile']} {summary_1['Maximum']} {summary_1['Standard Deviation']}", file=f)

        with open("charges_T2_tmp.txt", 'a') as f:
            print(f"From {init_time_hourly} to {new_time} {summary_2['Minimum']} {summary_2['0.05 quantile']} {summary_2['Median']} {summary_2['Mean']} {summary_2['0.95 quantile']} {summary_2['Maximum']} {summary_2['Standard Deviation']}", file=f)
        
        with open("charges_T3_tmp.txt", 'a') as f:
            print(f"From {init_time_hourly} to {new_time} {summary_3['Minimum']} {summary_3['0.05 quantile']} {summary_3['Median']} {summary_3['Mean']} {summary_3['0.95 quantile']} {summary_3['Maximum']} {summary_3['Standard Deviation']}", file=f)
        
        with open("charges_T4_tmp.txt", 'a') as f:
            print(f"From {init_time_hourly} to {new_time} {summary_4['Minimum']} {summary_4['0.05 quantile']} {summary_4['Median']} {summary_4['Mean']} {summary_4['0.95 quantile']} {summary_4['Maximum']} {summary_4['Standard Deviation']}", file=f)

        Q1_hourly = np.zeros(4)
        Q2_hourly = np.zeros(4)
        Q3_hourly = np.zeros(4)
        Q4_hourly = np.zeros(4)
        
        # Restart the clock ---------------------------------------------------
        init_time_hourly = new_time
        
        # End the hourly study ------------------------------------------------
    
    
    # -------------------------------------------------------------------------
    # Quarters magnitudes (every 15 minutes) ----------------------------------
    # -------------------------------------------------------------------------
    
    time_difference_quarters = (new_time - init_time_quarters).total_seconds()
    
    # Different time intervals can be considered
    if time_difference_quarters >= 900:
        
        print('--------------------------------------------------------------')
        print('15 minutes of events')
        print('--------------------------------------------------------------')
        
        time_interval = time_difference_quarters / 60
        
        # ---------------------------------------------------------------------
        # Mean multiplicities -------------------------------------------------
        # ---------------------------------------------------------------------
        
        # Calculate the figure of interest
        multiplicities_quarters = multiplicities_quarters[1:,:]
        
        mean_multiplicity_quarters = np.mean(multiplicities_quarters)
        
        mean_multiplicity_rpcwise_quarters = np.mean(multiplicities_quarters, axis = 0)
        # Create a mask to identify non-zero elements
        non_zero_mask = multiplicities_quarters != 0

        # Use the mask to calculate the mean for each column
        mean_multiplicity_non_zero_columns = np.nanmean(np.where(non_zero_mask, multiplicities_quarters, np.nan), axis=0)
        
        # 'mean_multiplicity_non_zero_columns' now contains the means for non-zero values column by column
        
        # If you want to ignore NaNs and get the mean for each column without non-zero elements:
        mean_multiplicity_non_zero_rpcwise_quarters = np.nanmean(np.where(non_zero_mask, multiplicities_quarters, np.nan), axis=0)
        
        non_zero_mask = multiplicities_quarters != 0

        # Use the mask to calculate the mean of all non-zero values in the matrix
        mean_multiplicity_non_zero_quarters = np.nanmean(np.where(non_zero_mask, multiplicities_quarters, np.nan))
        
        print(f'Mean multiplicities from {init_time}; \
Mean multiplicity in all layers, T1, T2, T3, T4; \
Removing zeroes in all layers, T1, T2, T3, T4;')
        print(f'{mean_multiplicity_quarters:.4g} \
{mean_multiplicity_rpcwise_quarters[0]:.4g} \
{mean_multiplicity_rpcwise_quarters[1]:.4g} \
{mean_multiplicity_rpcwise_quarters[2]:.4g} \
{mean_multiplicity_rpcwise_quarters[3]:.4g} \
{mean_multiplicity_non_zero_quarters:.4g} \
{mean_multiplicity_non_zero_rpcwise_quarters[0]:.4g} \
{mean_multiplicity_non_zero_rpcwise_quarters[1]:.4g} \
{mean_multiplicity_non_zero_rpcwise_quarters[2]:.4g} \
{mean_multiplicity_non_zero_rpcwise_quarters[3]:.4g}')
        
        with open("mean_multiplicities_tmp.txt", 'a') as f:
            print(f"From {init_time_quarters} to {new_time} \
{mean_multiplicity_quarters:.4g} \
{mean_multiplicity_rpcwise_quarters[0]:.4g} \
{mean_multiplicity_rpcwise_quarters[1]:.4g} \
{mean_multiplicity_rpcwise_quarters[2]:.4g} \
{mean_multiplicity_rpcwise_quarters[3]:.4g} \
{mean_multiplicity_non_zero_quarters:.4g} \
{mean_multiplicity_non_zero_rpcwise_quarters[0]:.4g} \
{mean_multiplicity_non_zero_rpcwise_quarters[1]:.4g} \
{mean_multiplicity_non_zero_rpcwise_quarters[2]:.4g} \
{mean_multiplicity_non_zero_rpcwise_quarters[3]:.4g}", file=f)
        
        # And we reset the variable
        multiplicities_quarters = np.zeros(4)
        
        # Restart the clock ---------------------------------------------------
        init_time_quarters = new_time
        
        # End the quarters study ----------------------------------------------
    

    # -------------------------------------------------------------------------
    # The part to stop the loop for debugging: in case limit = True -----------
    # -------------------------------------------------------------------------
    if limit:
        if i == limit_number:
            print('--------------------------------------------------------------')
            print("Cut by hand (not an error, see the code)")
            
            last_datetime_components = data.iloc[i, :6].astype(int)
            datetime_str = f"{last_datetime_components[0]:04d}-{last_datetime_components[1]:02d}-{last_datetime_components[2]:02d} {last_datetime_components[3]:02d}:{last_datetime_components[4]:02d}:{last_datetime_components[5]:02d}"
            last_datetime = pd.to_datetime(datetime_str, format="%Y-%m-%d %H:%M:%S")
            
            time_difference = (last_datetime - first_datetime).total_seconds()
            print(f' All the events were contained in {time_difference} seconds = {round(time_difference / 60)} min = {round(time_difference / 3600)} hours.')
            print('--------------------------------------------------------------')
            break
    # -------------------------------------------------------------------------
    
    total_TT1 = total_TT1 + 1
    
    # Convert the slices to NumPy arrays with the correct data type
    T1_F = T1_F_long[i, :]
    T1_B = T1_B_long[i, :]
    Q1_F = Q1_F_long[i, :]
    Q1_B = Q1_B_long[i, :]
    
    T2_F = T2_F_long[i, :]
    T2_B = T2_B_long[i, :]
    Q2_F = Q2_F_long[i, :]
    Q2_B = Q2_B_long[i, :]
    
    T3_F = T3_F_long[i, :]
    T3_B = T3_B_long[i, :]
    Q3_F = Q3_F_long[i, :]
    Q3_B = Q3_B_long[i, :]
    
    T4_F = T4_F_long[i, :]
    T4_B = T4_B_long[i, :]
    Q4_F = Q4_F_long[i, :]
    Q4_B = Q4_B_long[i, :]
    
    # -------------------------------------------------------------------------
    # Filtering and applying calibrations -------------------------------------
    # -------------------------------------------------------------------------
    
    # Time --------------------------------------------------------------------
    
    # We filter the values outside the TRB time window
    condition_met = (
        (T1_F > 0).any() or
        (T1_B > 0).any() or
        (T2_F > 0).any() or
        (T2_B > 0).any() or
        (T3_F > 0).any() or
        (T3_B > 0).any() or
        (T4_F > 0).any() or
        (T4_B > 0).any() or
        (T1_F < -200).any() or
        (T1_B < -200).any() or
        (T2_F < -200).any() or
        (T2_B < -200).any() or
        (T3_F < -200).any() or
        (T3_B < -200).any() or
        (T4_F < -200).any() or
        (T4_B < -200).any()
    )
    
    if condition_met:
        strip_time_fail += 1
        continue
    
    # Other filter
    condition_met = (
        (T1_F > 0).any() or
        (T1_B > 0).any() or
        (T2_F > 0).any() or
        (T2_B > 0).any() or
        (T3_F > 0).any() or
        (T3_B > 0).any() or
        (T4_F > 0).any() or
        (T4_B > 0).any() or
        (T1_F < -175).any() or
        (T1_B < -175).any() or
        (T2_F < -175).any() or
        (T2_B < -175).any() or
        (T3_F < -175).any() or
        (T3_B < -175).any() or
        (T4_F < -175).any() or
        (T4_B < -175).any()
    )
    
    if condition_met:
        strip_time_fail += 1
        continue
    
    T1 = T1_F - T1_B
    T2 = T2_F - T2_B
    T3 = T3_F - T3_B
    T4 = T4_F - T4_B
    
    # THIS CAN BE REWRITTEN SO IT COUNTS AND SHOWS
    # ERRORS IN DIFFERENT STRIP-SIDES, BUT FOR NOW IT WORKS (AND THE REWRITING
    # IS VERY EASY).
    
    # WE CAN STILL DO SOMETHING ELSE: IF ONE OF THE STRIPS DID NOT MEASURE IN
    # BUT OTHER STRIP MEASURED CORRECTLY IN BOTH SIDES WE COULD USE THAT VALUE,
    # WITH CERTAIN CARE.

    condition_met = (
        (T1 > 10).any() or
        (T2 > 10).any() or
        (T3 > 10).any() or
        (T4 > 10).any() or
        (T1 < - 10).any() or
        (T2 < - 10).any() or
        (T3 < - 10).any() or
        (T4 < - 10).any())
    
    if condition_met:
        strip_time_fail += 1
        continue
    
    T1_long = np.vstack((T1_long, T1))
    T2_long = np.vstack((T2_long, T2))
    T3_long = np.vstack((T3_long, T3))
    T4_long = np.vstack((T4_long, T4))
    
    # Time calibration:
    T1_cal = np.where(T1 != 0, T1 - calibration_T[0,:], 0)
    T2_cal = np.where(T2 != 0, T2 - calibration_T[1,:], 0)
    T3_cal = np.where(T3 != 0, T3 - calibration_T[2,:], 0)
    T4_cal = np.where(T4 != 0, T4 - calibration_T[3,:], 0)
    
    condition_met = (
        (T1_cal > 4).any() or
        (T2_cal > 4).any() or
        (T3_cal > 4).any() or
        (T4_cal > 4).any() or
        (T1_cal < -1).any() or
        (T2_cal < -1).any() or
        (T3_cal < -1).any() or
        (T4_cal < -1).any())
    
    if condition_met:
        strip_time_fail += 1
        continue
    
    # Charge ------------------------------------------------------------------
    
    # Average charge: if one of the values is zero then we take one of them.
    Q1 = calculate_charge_between_F_and_B(Q1_F, Q1_B)
    Q2 = calculate_charge_between_F_and_B(Q2_F, Q2_B)
    Q3 = calculate_charge_between_F_and_B(Q3_F, Q3_B)
    Q4 = calculate_charge_between_F_and_B(Q4_F, Q4_B)
    
    # We filter the values outside the TRB time window
    condition_met = (
        (Q1 > 300).any() or
        (Q2 > 300).any() or
        (Q3 > 300).any() or
        (Q4 > 300).any() or
        (Q1 < 0).any() or
        (Q2 < 0).any() or
        (Q3 < 0).any() or
        (Q4 < 0).any())
    
    if condition_met:
        strip_charge_fail += 1
        continue
    
    # Other filter
    condition_met = (
        (Q1 > 250).any() or
        (Q2 > 250).any() or
        (Q3 > 250).any() or
        (Q4 > 250).any() or
        (Q1 < 0).any() or
        (Q2 < 0).any() or
        (Q3 < 0).any() or
        (Q4 < 0).any())
    
    if condition_met:
        strip_charge_fail += 1
        continue
    
    # Charge calibration
    Q1_cal = np.where(Q1 != 0, Q1 - calibration_Q[0,:], 0)
    Q2_cal = np.where(Q2 != 0, Q2 - calibration_Q[1,:], 0)
    Q3_cal = np.where(Q3 != 0, Q3 - calibration_Q[2,:], 0)
    Q4_cal = np.where(Q4 != 0, Q4 - calibration_Q[3,:], 0)
    
    # We filter the values in a reasonable range once calibrated
    condition_met = (
        (Q1_cal > 200).any() or
        (Q2_cal > 200).any() or
        (Q3_cal > 200).any() or
        (Q4_cal > 200).any() or
        (Q1_cal < -5).any() or
        (Q2_cal < -5).any() or
        (Q3_cal < -5).any() or
        (Q4_cal < -5).any())
    
    if condition_met:
        strip_charge_fail += 1
        continue
    
    # -------------------------------------------------------------------------
    # And we store these calibrated values in the range of interest -----------
    # -------------------------------------------------------------------------
    
    T1_cal_long = np.vstack((T1_cal_long, T1_cal))
    T2_cal_long = np.vstack((T2_cal_long, T2_cal))
    T3_cal_long = np.vstack((T3_cal_long, T3_cal))
    T4_cal_long = np.vstack((T4_cal_long, T4_cal))
    
    Q1_cal_long = np.vstack((Q1_cal_long, Q1_cal))
    Q2_cal_long = np.vstack((Q2_cal_long, Q2_cal))
    Q3_cal_long = np.vstack((Q3_cal_long, Q3_cal))
    Q4_cal_long = np.vstack((Q4_cal_long, Q4_cal))
    
    # Original side to side values
    T1_F_filtered_long = np.vstack((T1_F_filtered_long, T1_cal))
    T1_B_filtered_long = np.vstack((T1_B_filtered_long, T1_cal))
    T2_F_filtered_long = np.vstack((T2_F_filtered_long, T2_cal))
    T2_B_filtered_long = np.vstack((T2_B_filtered_long, T2_cal))
    T3_F_filtered_long = np.vstack((T3_F_filtered_long, T3_cal))
    T3_B_filtered_long = np.vstack((T3_B_filtered_long, T3_cal))
    T4_F_filtered_long = np.vstack((T4_F_filtered_long, T4_cal))
    T4_B_filtered_long = np.vstack((T4_B_filtered_long, T4_cal))
    
    # Also we store the hourly charge
    Q1_hourly = np.vstack((Q1_hourly, Q1_cal))
    Q2_hourly = np.vstack((Q2_hourly, Q2_cal))
    Q3_hourly = np.vstack((Q3_hourly, Q3_cal))
    Q4_hourly = np.vstack((Q4_hourly, Q4_cal))
    
    # Count the number of events that reach this part of the code
    filtered_events += 1    
    
    # -------------------------------------------------------------------------
    # CHARGE STUDY ------------------------------------------------------------
    # -------------------------------------------------------------------------
    
    # In only one strip
    condition_met = (np.count_nonzero(Q1_cal) == 1)
    if condition_met:
        Q1_ind_strip = np.vstack((Q1_ind_strip, Q1_cal))
        
    condition_met = (np.count_nonzero(Q2_cal) == 1)
    if condition_met:
        Q2_ind_strip = np.vstack((Q2_ind_strip, Q2_cal))
        
    condition_met = (np.count_nonzero(Q3_cal) == 1)
    if condition_met:
        Q3_ind_strip = np.vstack((Q3_ind_strip, Q3_cal))
        
    condition_met = (np.count_nonzero(Q4_cal) == 1)
    if condition_met:
        Q4_ind_strip = np.vstack((Q4_ind_strip, Q4_cal))
    
    # In two strips
    condition_met = (np.count_nonzero(Q1_cal) == 2)
    if condition_met:
        Q1_double_strip = np.vstack((Q1_double_strip, Q1_cal))
        if interstrip(Q1_cal):
            Q1_interstrip = np.vstack((Q1_interstrip, Q1_cal))
        
    condition_met = (np.count_nonzero(Q2_cal) == 2)
    if condition_met:
        Q2_double_strip = np.vstack((Q2_double_strip, Q2_cal))
        if interstrip(Q2_cal):
            Q2_interstrip = np.vstack((Q2_interstrip, Q2_cal))
        
    condition_met = (np.count_nonzero(Q3_cal) == 2)
    if condition_met:
        Q3_double_strip = np.vstack((Q3_double_strip, Q3_cal))
        if interstrip(Q3_cal):
            Q3_interstrip = np.vstack((Q3_interstrip, Q3_cal))
            
    condition_met = (np.count_nonzero(Q4_cal) == 2)
    if condition_met:
        Q4_double_strip = np.vstack((Q4_double_strip, Q4_cal))
        if interstrip(Q4_cal):
            Q4_interstrip = np.vstack((Q4_interstrip, Q4_cal))
            
    # In three strips.
    condition_met = (np.count_nonzero(Q1_cal) == 3)
    if condition_met:
        Q1_triple_strip = np.vstack((Q1_triple_strip, Q1_cal))
        
    condition_met = (np.count_nonzero(Q2_cal) == 3)
    if condition_met:
        Q2_triple_strip = np.vstack((Q2_triple_strip, Q2_cal))
        
    condition_met = (np.count_nonzero(Q3_cal) == 3)
    if condition_met:
        Q3_triple_strip = np.vstack((Q3_triple_strip, Q3_cal))
        
    condition_met = (np.count_nonzero(Q4_cal) == 3)
    if condition_met:
        Q4_triple_strip = np.vstack((Q4_triple_strip, Q4_cal))
    
    # In four strips.
    condition_met = (np.count_nonzero(Q1_cal) == 4)
    if condition_met:
        Q1_quad_strip = np.vstack((Q1_quad_strip, Q1_cal))
        
    condition_met = (np.count_nonzero(Q2_cal) == 4)
    if condition_met:
        Q2_quad_strip = np.vstack((Q2_quad_strip, Q2_cal))
        
    condition_met = (np.count_nonzero(Q3_cal) == 4)
    if condition_met:
        Q3_quad_strip = np.vstack((Q3_quad_strip, Q3_cal))
        
    condition_met = (np.count_nonzero(Q4_cal) == 4)
    if condition_met:
        Q4_quad_strip = np.vstack((Q4_quad_strip, Q4_cal))
    
    # All the charge is stored.
    Q1_strip = np.vstack((Q1_strip, Q1_cal))
    Q2_strip = np.vstack((Q2_strip, Q2_cal))
    Q3_strip = np.vstack((Q3_strip, Q3_cal))
    Q4_strip = np.vstack((Q4_strip, Q4_cal))
    
    # -------------------------------------------------------------------------
    # Multiplicity study ------------------------------------------------------
    # -------------------------------------------------------------------------
    
    multi[0] = multiplicity(T1_cal, Q1_cal)
    multi[1] = multiplicity(T2_cal, Q2_cal)
    multi[2] = multiplicity(T3_cal, Q3_cal)
    multi[3] = multiplicity(T4_cal, Q4_cal)
    
    multiplicities = np.vstack((multiplicities, multi))
    multiplicities_quarters = np.vstack((multiplicities_quarters, multi))
    
    # In only one strip
    condition_met = (multi[0] == 0)
    if condition_met:
        Q1_n0_strip = np.vstack((Q1_n1_strip, Q1_cal))
        
    condition_met = (multi[1] == 0)
    if condition_met:
        Q2_n0_strip = np.vstack((Q2_n1_strip, Q2_cal))
        
    condition_met = (multi[2] == 0)
    if condition_met:
        Q3_n0_strip = np.vstack((Q3_n1_strip, Q3_cal))
        
    condition_met = (multi[3] == 0)
    if condition_met:
        Q4_n0_strip = np.vstack((Q4_n1_strip, Q4_cal))
    
    # In only one strip
    condition_met = (multi[0] == 1)
    if condition_met:
        Q1_n1_strip = np.vstack((Q1_n1_strip, Q1_cal))
        
    condition_met = (multi[1] == 1)
    if condition_met:
        Q2_n1_strip = np.vstack((Q2_n1_strip, Q2_cal))
        
    condition_met = (multi[2] == 1)
    if condition_met:
        Q3_n1_strip = np.vstack((Q3_n1_strip, Q3_cal))
        
    condition_met = (multi[3] == 1)
    if condition_met:
        Q4_n1_strip = np.vstack((Q4_n1_strip, Q4_cal))
    
    # In two strips: note that those have to be two consecutive indexes.
    condition_met = (multi[0] == 2)
    if condition_met:
        Q1_n2_strip = np.vstack((Q1_n2_strip, Q1_cal))
    
    condition_met = (multi[1] == 2)
    if condition_met:
        Q2_n2_strip = np.vstack((Q2_n2_strip, Q2_cal))
        
    condition_met = (multi[2] == 2)
    if condition_met:
        Q3_n2_strip = np.vstack((Q3_n2_strip, Q3_cal))
    
    condition_met = (multi[3] == 2)
    if condition_met:
        Q4_n2_strip = np.vstack((Q4_n2_strip, Q4_cal))
    
    # In three strips: note that those have to be three consecutive indexes.
    condition_met = (multi[0] == 3)
    if condition_met:
        Q1_n3_strip = np.vstack((Q1_n3_strip, Q1_cal))
        
    condition_met = (multi[1] == 3)
    if condition_met:
        Q2_n3_strip = np.vstack((Q2_n3_strip, Q2_cal))
        
    condition_met = (multi[2] == 3)
    if condition_met:
        Q3_n3_strip = np.vstack((Q3_n3_strip, Q3_cal))
        
    condition_met = (multi[3] == 3)
    if condition_met:
        Q4_n3_strip = np.vstack((Q4_n3_strip, Q4_cal))
    
    # In four strips.
    condition_met = (multi[0] == 4)
    if condition_met:
        Q1_n4_strip = np.vstack((Q1_n4_strip, Q1_cal))
        
    condition_met = (multi[1] == 4)
    if condition_met:
        Q2_n4_strip = np.vstack((Q2_n4_strip, Q2_cal))
        
    condition_met = (multi[2] == 4)
    if condition_met:
        Q3_n4_strip = np.vstack((Q3_n4_strip, Q3_cal))
        
    condition_met = (multi[3] == 4)
    if condition_met:
        Q4_n4_strip = np.vstack((Q4_n4_strip, Q4_cal))
    
    half_tapped_events += 1
    
    # -------------------------------------------------------------------------
    # Positions ---------------------------------------------------------------
    # -------------------------------------------------------------------------
    
    # Get position from times. All in mm. Functions in header.
    
    if multi[0] <= 1 and multi[1] <= 1 and multi[2] <= 1 and multi[3] <= 1:
        pos_1 = time_to_position(T1_cal, Q1_cal, 4)
        pos_2 = time_to_position(T2_cal, Q2_cal, 1)
        pos_3 = time_to_position(T3_cal, Q3_cal, 4)
        pos_4 = time_to_position(T4_cal, Q4_cal, 1)
    else:
        position_not_asigned += 1
        continue
    
    strongly_filtered_events += 1
    
    # Create the 3D point
    pos_1_3d = np.concatenate((pos_1, np.array([0])))
    pos_2_3d = np.concatenate((pos_2, np.array([100])))
    pos_3_3d = np.concatenate((pos_3, np.array([200])))
    pos_4_3d = np.concatenate((pos_4, np.array([400])))
    
    positions_to_stack = [pos_1_3d, pos_2_3d, pos_3_3d, pos_4_3d]
    
    # Filter out arrays with non-negative components
    filtered_positions = [pos for pos in positions_to_stack if (pos >= 0).all()]
    
    # Convert filtered positions to a NumPy array and stack them
    if filtered_positions:
        stacked_positions = np.vstack(filtered_positions)
    
    positions = np.vstack((positions, stacked_positions))
    
    # -------------------------------------------------------------------------
    # Filter the multiplicity n = 1 to check for muons or energ. electrons ----
    # -------------------------------------------------------------------------
    
    # TO REALLY GET N=1 WE SHOULD CHECK THE CHARGE TO SEE IF THERE ARE
    # STREAMERS OR ONE STRIP MULTIHITS.
    
    # -------------------------------------------------------------------------
    # T1. WE NEED GEOMETRIC STUDY ---------------------------------------------
    # -------------------------------------------------------------------------
    
    # 0. Check if the hit has multiplicity n = 1
    n1_condition = multi[0] <= 1
    
    # 1. Check if there are hits in the three other planes.
    time_condition_met = (pos_2_3d[0] > 0 and \
                        pos_3_3d[0] > 0 and \
                        pos_4_3d[0] > 0)
        
    # 2. Check if the three points studied are aligned.
    line_condition = are_points_on_same_line(pos_2_3d, pos_3_3d, pos_4_3d)
    
    # 3. Check if the line traced by the points passes through the RPC studied.
    plane_z = 0
    plane_condition_met = (is_line_through_plane(pos_2_3d, pos_3_3d, plane_z) and \
                            is_line_through_plane(pos_2_3d, pos_4_3d, plane_z) and \
                            is_line_through_plane(pos_3_3d, pos_4_3d, plane_z))
    
    total_muons_1 += 1
    if plane_condition_met and time_condition_met and line_condition and n1_condition:
        if plot_trajectories: plot_3d_scene(pos_2_3d, pos_3_3d, pos_4_3d, plane_z)
        
        total_crossing_muons_1 += 1
        hourly_total_crossing_muons_1 += 1
        if pos_1_3d[0] > 0:
            line_condition_a = are_points_on_same_line(pos_1_3d, pos_3_3d, pos_4_3d)
            line_condition_b = are_points_on_same_line(pos_1_3d, pos_2_3d, pos_3_3d)
            line_condition_c = are_points_on_same_line(pos_1_3d, pos_2_3d, pos_4_3d)
            
            if (line_condition_a or line_condition_b or line_condition_c) and multi[0] == 1:
                detected_1 +=1
                hourly_detected_1 +=1
                
                positions_detected = np.vstack((positions_detected, pos_1_3d))
            else:
                positions_non_detected = np.vstack((positions_non_detected, pos_1_3d))

    # -------------------------------------------------------------------------
    # T2 ----------------------------------------------------------------------
    # -------------------------------------------------------------------------
    
    # 0. Check if the hit has multiplicity n = 1
    n1_condition = multi[1] <= 1
    
    # 1. Check if there are hits in the three other planes.
    time_condition_met = (pos_1_3d[0] > 0 and \
                        pos_3_3d[0] > 0 and \
                        pos_4_3d[0] > 0)
    
    # 2. Check if the three points studied are aligned.
    line_condition = are_points_on_same_line(pos_1_3d, pos_3_3d, pos_4_3d)
    
    # 3. Check if the line traced by the points passes through the RPC studied.
    plane_z = 100
    plane_condition_met = (is_line_through_plane(pos_1_3d, pos_3_3d, plane_z) and \
                            is_line_through_plane(pos_1_3d, pos_4_3d, plane_z) and \
                            is_line_through_plane(pos_3_3d, pos_4_3d, plane_z))
    
    total_muons_2 += 1
    if plane_condition_met and time_condition_met and line_condition and n1_condition:
        if plot_trajectories: plot_3d_scene(pos_1_3d, pos_3_3d, pos_4_3d, plane_z)
        
        total_crossing_muons_2 += 1
        hourly_total_crossing_muons_2 += 1
        if pos_2_3d[0] > 0:
            line_condition_a = are_points_on_same_line(pos_2_3d, pos_3_3d, pos_4_3d)
            line_condition_b = are_points_on_same_line(pos_2_3d, pos_1_3d, pos_3_3d)
            line_condition_c = are_points_on_same_line(pos_2_3d, pos_1_3d, pos_4_3d)
            
            if (line_condition_a or line_condition_b or line_condition_c) and multi[1] == 1:
                detected_2 += 1
                hourly_detected_2 += 1
                
                positions_detected = np.vstack((positions_detected, pos_2_3d))
            else:
                positions_non_detected = np.vstack((positions_non_detected, pos_2_3d))
    
    # -------------------------------------------------------------------------
    # T3 ----------------------------------------------------------------------
    # -------------------------------------------------------------------------
    
    # 0. Check if the hit has multiplicity n = 1
    n1_condition = multi[2] <= 1
    
    # 1. Check if there are hits in the three other planes.
    time_condition_met = (pos_1_3d[0] > 0 and \
                        pos_2_3d[0] > 0 and \
                        pos_4_3d[0] > 0)
    
    # 2. Check if the three points studied are aligned.
    line_condition = are_points_on_same_line(pos_1_3d, pos_2_3d, pos_4_3d)

    # 3. Check if the line traced by the points passes through the RPC studied.
    plane_z = 200
    plane_condition_met = (is_line_through_plane(pos_1_3d, pos_2_3d, plane_z) and \
                            is_line_through_plane(pos_2_3d, pos_4_3d, plane_z) and \
                            is_line_through_plane(pos_1_3d, pos_4_3d, plane_z))
    
    total_muons_3 += 1
    if plane_condition_met and time_condition_met and line_condition and n1_condition:
        if plot_trajectories: plot_3d_scene(pos_1_3d, pos_2_3d, pos_4_3d, plane_z)
        
        total_crossing_muons_3 += 1
        hourly_total_crossing_muons_3 += 1
        if pos_3_3d[0] > 0:
            line_condition_a = are_points_on_same_line(pos_3_3d, pos_1_3d, pos_4_3d)
            line_condition_b = are_points_on_same_line(pos_3_3d, pos_1_3d, pos_2_3d)
            line_condition_c = are_points_on_same_line(pos_3_3d, pos_2_3d, pos_4_3d)
            
            if (line_condition_a or line_condition_b or line_condition_c) and multi[2] == 1:
                detected_3 += 1
                hourly_detected_3 += 1
                
                positions_detected = np.vstack((positions_detected, pos_3_3d))
            else:
                positions_non_detected = np.vstack((positions_non_detected, pos_3_3d))
    
    # -------------------------------------------------------------------------
    # T4. WE NEED GEOMETRIC STUDY ---------------------------------------------
    # -------------------------------------------------------------------------
    
    # 0. Check if the hit has multiplicity n = 1
    n1_condition = multi[3] <= 1
    
    # 1. Check if there are hits in the three other planes.
    time_condition_met = (pos_1_3d[0] > 0 and \
                        pos_2_3d[0] > 0 and \
                        pos_3_3d[0] > 0)
        
    # 2. Check if the three points studied are aligned.
    line_condition = are_points_on_same_line(pos_1_3d, pos_2_3d, pos_3_3d)
    
    # 3. Check if the line traced by the points passes through the RPC studied.
    plane_z = 400
    plane_condition_met= (is_line_through_plane(pos_1_3d, pos_2_3d, plane_z) and \
                            is_line_through_plane(pos_2_3d, pos_3_3d, plane_z) and \
                            is_line_through_plane(pos_1_3d, pos_3_3d, plane_z))
    
    total_muons_4 += 1
    if plane_condition_met and time_condition_met and line_condition and n1_condition:
        if plot_trajectories: plot_3d_scene(pos_1_3d, pos_2_3d, pos_3_3d, plane_z)
        
        total_crossing_muons_4 += 1
        hourly_total_crossing_muons_4 +=1
        if pos_4_3d[0] > 0:
            line_condition_a = are_points_on_same_line(pos_4_3d, pos_1_3d, pos_3_3d)
            line_condition_b = are_points_on_same_line(pos_4_3d, pos_2_3d, pos_3_3d)
            line_condition_c = are_points_on_same_line(pos_4_3d, pos_1_3d, pos_2_3d)
            
            if (line_condition_a or line_condition_b or line_condition_c) and multi[3] == 1:
                detected_4 += 1
                hourly_detected_4 += 1
                
                positions_detected = np.vstack((positions_detected, pos_4_3d))
            else:
                positions_non_detected = np.vstack((positions_non_detected, pos_4_3d))

    fully_tapped_events += 1

    # -------------------------------------------------------------------------
    # Naive intrinsic eff. calculation ----------------------------------------
    # -------------------------------------------------------------------------
    
    # T1
    time_condition_met = (pos_2_3d[0] > 0 and \
                        pos_3_3d[0] > 0 and \
                        pos_4_3d[0] > 0)
    
    if time_condition_met:
        total_crossing_muons_1_naive += 1
        if pos_1_3d[0] > 0:
            detected_1_naive +=1
            
    # T2
    time_condition_met = (pos_1_3d[0] > 0 and \
                        pos_3_3d[0] > 0 and \
                        pos_4_3d[0] > 0)
    
    if time_condition_met:
        total_crossing_muons_2_naive += 1
        if pos_2_3d[0] > 0:
            detected_2_naive +=1
            
    # T3
    time_condition_met = (pos_2_3d[0] > 0 and \
                        pos_1_3d[0] > 0 and \
                        pos_4_3d[0] > 0)
    
    if time_condition_met:
        total_crossing_muons_3_naive += 1
        if pos_3_3d[0] > 0:
            detected_3_naive +=1
            
    # T4
    time_condition_met = (pos_2_3d[0] > 0 and \
                        pos_3_3d[0] > 0 and \
                        pos_1_3d[0] > 0)
    
    if time_condition_met:
        total_crossing_muons_4_naive += 1
        if pos_4_3d[0] > 0:
            detected_4_naive +=1
    
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Outside the loop ------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

# Total time of events calculation
if limit == False:
    print('--------------------------------------------------------------')
    last_index = total_number_of_events - 1

    last_datetime_components = data.iloc[last_index, :6].astype(int)
    datetime_str = f"{last_datetime_components[0]:04d}-{last_datetime_components[1]:02d}-{last_datetime_components[2]:02d} {last_datetime_components[3]:02d}:{last_datetime_components[4]:02d}:{last_datetime_components[5]:02d}"
    last_datetime = pd.to_datetime(datetime_str, format="%Y-%m-%d %H:%M:%S")
    
    time_difference = (last_datetime - first_datetime).total_seconds()
    print(f' All the events were contained in {time_difference} seconds = {round(time_difference / 60)} min = {round(time_difference / 3600)} hours.')
    print('--------------------------------------------------------------')
    
# -----------------------------------------------------------------------------
# CLean the data: removing the first row (that is of zeroes) ------------------
# -----------------------------------------------------------------------------

# T1_long = T1_long[1:, :]
# T2_long = T2_long[1:, :]
# T3_long = T3_long[1:, :]
# T4_long = T4_long[1:, :]

Q1_double_strip = Q1_double_strip[1:, :]
Q2_double_strip = Q2_double_strip[1:, :]
Q3_double_strip = Q3_double_strip[1:, :]
Q4_double_strip = Q4_double_strip[1:, :]


# -----------------------------------------------------------------------------
# Some result calculations ----------------------------------------------------
# -----------------------------------------------------------------------------

eff1 = detected_1 / total_crossing_muons_1
eff2 = detected_2 / total_crossing_muons_2
eff3 = detected_3 / total_crossing_muons_3
eff4 = detected_4 / total_crossing_muons_4

eff1_naive = detected_1_naive / total_crossing_muons_1_naive
eff2_naive = detected_2_naive / total_crossing_muons_2_naive
eff3_naive = detected_3_naive / total_crossing_muons_3_naive
eff4_naive = detected_4_naive / total_crossing_muons_4_naive

geo_eff1 = total_crossing_muons_1 / total_muons_1
geo_eff2 = total_crossing_muons_2 / total_muons_2
geo_eff3 = total_crossing_muons_3 / total_muons_3
geo_eff4 = total_crossing_muons_4 / total_muons_4

efficiencies = np.array([eff1, eff2, eff3, eff4])
efficiencies_naive = np.array([eff1_naive, eff2_naive, eff3_naive, eff4_naive])
geo_efficiencies = np.array([geo_eff1, geo_eff2, geo_eff3, geo_eff4])
detected_counts = np.array([detected_1, detected_2, detected_3, detected_4])
total_crossing_muons = np.array([total_crossing_muons_1, total_crossing_muons_2, total_crossing_muons_3, total_crossing_muons_4])

# Start and end times in a proper format
start = first_datetime.isoformat()
start = start.replace('T', ' ')

end = last_datetime.isoformat()
end = end.replace('T', ' ')

# -----------------------------------------------------------------------------
# Going back to the results directory -----------------------------------------
# -----------------------------------------------------------------------------

os.chdir("..")

if os.path.exists(f"DAQ_data_analysis_results_from_{start}_to_{end}"):
    # Create the directory
    shutil.rmtree(f"DAQ_data_analysis_results_from_{start}_to_{end}")

shutil.copytree("DAQ_data_analysis_results_tmp", f"DAQ_data_analysis_results_from_{start}_to_{end}")
shutil.rmtree("DAQ_data_analysis_results_tmp")

os.chdir(f"DAQ_data_analysis_results_from_{start}_to_{end}")

# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Output file -----------------------------------------------------------------
# -----------------------------------------------------------------------------

with open(f'output_from_{start}_to_{end}.txt', 'w') as f:
    print('----------------------------------------------------------------------', file=f)
    print(f' Results from {start} to {end}', file=f)

with open(f'output_from_{start}_to_{end}.txt', 'a') as f:
    print('----------------------------------------------------------------------', file=f)
    print(f' All the events were contained in {time_difference} seconds \
= {round(time_difference / 60)} min = {round(time_difference / 3600)} hours.', file=f)
    print('----------------------------------------------------------------------', file=f)
    print("Layers:                  T1    T2    T3    T4", file=f)
    print("Intrinsic eff:          ", np.round(efficiencies*100)/100, file=f)
    print("Naive intrinsic eff:    ", np.round(efficiencies_naive*100)/100, file=f)
    print("Geometric eff:          ", np.round(geo_efficiencies*100)/100, file=f)
    print("Detected counts:    ", detected_counts, file=f)
    print('\n', file=f)
    print('Clean n=1 events per RPC used to calculate efficiency:', file=f)
    print("\tPer RPC:", total_crossing_muons, file=f)
    print("\tIn mingo:", sum(total_crossing_muons), file=f)
    print('\n', file=f)
    print("Total original events:", total_number_of_events, file=f)
    print("\tTotal TT1 (three-plane coincidence) events:", total_TT1, file=f)
    print("\tTotal TT2 (self-trigger) events:", total_TT2, file=f)
    print('\n', file=f)
    print(f"Number of events that are correctly registered {filtered_events}", file=f)
    print(f"Number of events tapped to calculate eficiency {fully_tapped_events}", file=f)
    print(f"Ratio of data tapped from the total to calculate eff: {round(np.array(sum(total_crossing_muons) / filtered_events) * 100, 1)} %", file=f)
    print(f"Ratio of data tapped from the real events to calculate eff: {round(np.array(sum(total_crossing_muons) / total_TT1) * 100, 1)} %", file=f)
    print('\n', file=f)
    print("Events in which some strip failed:", file=f)
    print("\tIn time:", strip_time_fail, file=f)
    print("\tIn charge:", strip_charge_fail, file=f)
    print('----------------------------------------------------------------------', file=f)


# -----------------------------------------------------------------------------
# Hourly files ----------------------------------------------------------------
# -----------------------------------------------------------------------------

# Rates file
os.rename("rates_tmp.txt", f"rates_from_{init_time}_to_{end}.txt")

# Efficiencies file
os.rename("efficiencies_tmp.txt", f"efficiencies_from_{init_time}_to_{end}.txt")

# Charges file
os.rename("charges_T1_tmp.txt", f"charges_in_T1_from_{init_time}_to_{end}.txt")
os.rename("charges_T2_tmp.txt", f"charges_in_T2_from_{init_time}_to_{end}.txt")
os.rename("charges_T3_tmp.txt", f"charges_in_T3_from_{init_time}_to_{end}.txt")
os.rename("charges_T4_tmp.txt", f"charges_in_T4_from_{init_time}_to_{end}.txt")
os.rename("mean_multiplicities_tmp.txt", f"mean_multiplicities_from_{init_time}_to_{end}.txt")


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Charge study ----------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Slewing correction variables ------------------------------------------------
# -----------------------------------------------------------------------------

# Variables
title = "Slewing effect"
x = Q1_cal_long[:,0]
y = T1_cal_long[:,0] - T2_B_filtered_long[:,0]
original_string = title
filename = 'slewing_correction'


# Execution
# Create a 2D histogram to calculate point density
mask = (x != 0) & (y != 0)
x = x[mask]
y = y[mask]

# Create a histogram of negative values
heatmap, xedges, yedges = np.histogram2d(x, y, bins=100)
heatmap = np.log(heatmap + 1)  # Adding 1 to avoid log(0)
plt.close()

fig, ax = plt.subplots(figsize=(12, 4))
cax = ax.pcolormesh(xedges, yedges, heatmap.T, cmap='viridis')
plt.colorbar(cax, label='Log Density')
dated_title = f'{title} at {first_datetime}'
plt.title(dated_title)
plt.grid(True)
plt.tight_layout()
ax.set_xlabel("Charge (AU, ns)")
ax.set_ylabel("Time difference between calibrated time and another RPC strip (ns)")

# ax.set_xlim([0, 200])
# ax.set_ylim([-3, 3])

plt.savefig(f'{filename}.png', format="png")
if show_plots: plt.show(); plt.close()


plt.close()
fig = plt.figure(figsize=(10,7))
plt.scatter(x, y)
plt.ylabel('Charge (AU)')
plt.xlabel('Position (ns)')
plt.title(f'Charge per strip for events \nfrom {start} to {end}')
plt.xlim([0, 80])
plt.ylim([-7, 9])
plt.grid(True)

plt.tight_layout()
if show_plots: plt.show(); plt.close()

# -----------------------------------------------------------------------------
# The charge according to the position ----------------------------------------
# -----------------------------------------------------------------------------

title = "Title"
x = T1_cal_long[:,1]
y = Q1_cal_long[:,1]
original_string = title
new_title = '{}_s{}'.format(original_string[0:2], original_string[-1])
filename = f'{first_datetime}_{new_title}_diagonal_histogram'

# Create a 2D histogram to calculate point density
mask = (x != 0) & (y != 0)
# Apply the mask to both x and y
x = x[mask]
y = y[mask]
# Create a histogram of negative values
heatmap, xedges, yedges = np.histogram2d(x, y, bins=100)
# Apply a logarithmic transformation to the density values
heatmap = np.log(heatmap + 1)  # Adding 1 to avoid log(0)
plt.close()
# Create a figure and axis for the plot
fig, ax = plt.subplots(figsize=(16, 10))
# Use Matplotlib's pcolormesh to create a heatmap with log-scaled colors
cax = ax.pcolormesh(xedges, yedges, heatmap.T, cmap='viridis')
# Add a colorbar to the plot
plt.colorbar(cax, label='Log Density')
# Set axis labels and title
ax.set_title('Log-Scaled 2D Point Density Plot')
dated_title = f'{title} at {first_datetime}'
plt.title(dated_title)
# Add a legend
plt.grid(True)
plt.tight_layout()
# SET THE FORMAT IMAGE TO PNG
plt.savefig(f'{output_order}_{filename}.png', format="png")
output_order = output_order + 1
# Display the plot
if show_plots: plt.show(); plt.close()


# plt.close()
# fig = plt.figure(figsize=(10,7))
# plt.scatter(T1_cal_long, Q1_cal_long)
# plt.xlabel('Position (ns)')
# plt.ylabel('Charge (AU)')
# plt.title(f'Charge per strip for events \nfrom {start} to {end}')
# # plt.xlim([-70, x_axes_limit_plots])
# # plt.ylim([None, 30000])
# # plt.grid(True)

# plt.tight_layout()
# if show_plots: plt.show(); plt.close()

# -----------------------------------------------------------------------------
# Preamble --------------------------------------------------------------------
# -----------------------------------------------------------------------------
normalized_condition = False

# The figsize
v = (5, 4)

# -----------------------------------------------------------------------------
# Charge calibration check ----------------------------------------------------
# -----------------------------------------------------------------------------

Q_strip = np.vstack((Q1_strip, Q2_strip, Q3_strip, Q4_strip))

# Study of the negative charges
matrix = Q_strip
negative_values = matrix.flatten()[matrix.flatten() != 0]

plt.close()
fig = plt.figure(figsize=v)

# Create a histogram of negative values
plt.hist(negative_values, bins='auto', color='red', alpha=0.7)
plt.xlabel('Charge (AU)')
plt.ylabel('Counts')
plt.title(f'Charge per strip for {len(negative_values)} events \nfrom {start} to {end}')
plt.xlim([-70, x_axes_limit_plots])
# plt.ylim([None, 30000])
# plt.grid(True)

plt.tight_layout()
if show_plots: plt.show(); plt.close()


# -----------------------------------------------------------------------------
# Charge in each strip --------------------------------------------------------
# -----------------------------------------------------------------------------

def multiplot(RPC1, RPC2, RPC3, RPC4, my_title, normalized, stripwise, summed, rpcwise, name):
    global output_order
    
    name = f'{start}_{name}'
    new_title = f'{my_title} \nfrom {start} to {end}'
    
    if stripwise:
        filename = f'{name}_stripwise'
        
        data1_0 = RPC1[:,0]
        data1_1 = RPC1[:,1]
        data1_2 = RPC1[:,2]
        data1_3 = RPC1[:,3]
        
        data2_0 = RPC2[:,0]
        data2_1 = RPC2[:,1]
        data2_2 = RPC2[:,2]
        data2_3 = RPC2[:,3]
        
        data3_0 = RPC3[:,0]
        data3_1 = RPC3[:,1]
        data3_2 = RPC3[:,2]
        data3_3 = RPC3[:,3]
        
        data4_0 = RPC4[:,0]
        data4_1 = RPC4[:,1]
        data4_2 = RPC4[:,2]
        data4_3 = RPC4[:,3]
        
        # Apply mask to remove zeroes
        mask = data1_0 != 0
        data1_0 = data1_0[mask]
        mask = data1_1 != 0
        data1_1 = data1_1[mask]
        mask = data1_2 != 0
        data1_2 = data1_2[mask]
        mask = data1_3 != 0
        data1_3 = data1_3[mask]
        mask = data2_0 != 0
        data2_0 = data2_0[mask]
        mask = data2_1 != 0
        data2_1 = data2_1[mask]
        mask = data2_2 != 0
        data2_2 = data2_2[mask]
        mask = data2_3 != 0
        data2_3 = data2_3[mask]
        mask = data3_0 != 0
        data3_0 = data3_0[mask]
        mask = data3_1 != 0
        data3_1 = data3_1[mask]
        mask = data3_2 != 0
        data3_2 = data3_2[mask]
        mask = data3_3 != 0
        data3_3 = data3_3[mask]
        mask = data4_0 != 0
        data4_0 = data4_0[mask]
        mask = data4_1 != 0
        data4_1 = data4_1[mask]
        mask = data4_2 != 0
        data4_2 = data4_2[mask]
        mask = data4_3 != 0
        data4_3 = data4_3[mask]
        
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))
        
        # # Maybe a different bin width is required
        # if np.count_nonzero(RPC1[5,:]) == 3 and len(RPC1[:,0]) < 10000:
        #     bin_num = 1000
        # else:
        #     bin_num = 'auto'
        
        bin_num = 'auto'
        
        # Plot histograms on each subplot
        axes[0, 0].hist(data1_0, bins=bin_num, alpha=0.5, color='blue', label=f'Strip 1, {len(data1_0)} events', density = normalized)
        axes[0, 0].hist(data1_1, bins=bin_num, alpha=0.5, color='orange', label=f'Strip 2, {len(data1_1)} events', density = normalized)
        axes[0, 0].hist(data1_2, bins=bin_num, alpha=0.5, color='red', label=f'Strip 3, {len(data1_2)} events', density = normalized)
        axes[0, 0].hist(data1_3, bins=bin_num, alpha=0.5, color='green', label=f'Strip 4, {len(data1_3)} events', density = normalized)
        axes[0, 0].legend()
        axes[0, 0].set_title('T1, z = 0')
        
        # Plot histograms on other subplots using the same pattern
        axes[0, 1].hist(data2_0, bins=bin_num, alpha=0.5, color='blue', label=f'Strip 1, {len(data2_0)} events', density = normalized)
        axes[0, 1].hist(data2_1, bins=bin_num, alpha=0.5, color='orange', label=f'Strip 2, {len(data2_1)} events', density = normalized)
        axes[0, 1].hist(data2_2, bins=bin_num, alpha=0.5, color='red', label=f'Strip 3, {len(data2_2)} events', density = normalized)
        axes[0, 1].hist(data2_3, bins=bin_num, alpha=0.5, color='green', label=f'Strip 4, {len(data2_3)} events', density = normalized)
        axes[0, 1].legend()
        axes[0, 1].set_title('T2, z = 100')
        
        axes[1, 0].hist(data3_0, bins=bin_num, alpha=0.5, color='blue', label=f'Strip 1, {len(data3_0)} events', density = normalized)
        axes[1, 0].hist(data3_1, bins=bin_num, alpha=0.5, color='orange', label=f'Strip 2, {len(data3_1)} events', density = normalized)
        axes[1, 0].hist(data3_2, bins=bin_num, alpha=0.5, color='red', label=f'Strip 3, {len(data3_2)} events', density = normalized)
        axes[1, 0].hist(data3_3, bins=bin_num, alpha=0.5, color='green', label=f'Strip 4, {len(data3_3)} events', density = normalized)
        axes[1, 0].legend()
        axes[1, 0].set_title('T3, z = 200')
        
        axes[1, 1].hist(data4_0, bins=bin_num, alpha=0.5, color='blue', label=f'Strip 1, {len(data4_0)} events', density = normalized)
        axes[1, 1].hist(data4_1, bins=bin_num, alpha=0.5, color='orange', label=f'Strip 2, {len(data4_1)} events', density = normalized)
        axes[1, 1].hist(data4_2, bins=bin_num, alpha=0.5, color='red', label=f'Strip 3, {len(data4_2)} events', density = normalized)
        axes[1, 1].hist(data4_3, bins=bin_num, alpha=0.5, color='green', label=f'Strip 4, {len(data4_3)} events', density = normalized)
        axes[1, 1].legend()
        axes[1, 1].set_title('T4, z = 400')
        
        for ax in axes.flatten():
            ax.set_xlabel('Charge (AU)')
            ax.set_ylabel('Counts')
            ax.set_xlim(-5, x_axes_limit_plots)  # The second argument (None) means "beyond"
        
        # # Maybe it is interesting to focus on a certain region
        # if np.count_nonzero(RPC1[5,:]) == 3:
        #     for ax in axes.flatten():
        #         ax.set_xlim(-1, 2.3)  # The second argument (None) means "beyond"
        
        plt.suptitle(f'{new_title}', fontsize=16)
        plt.tight_layout()
        
    else:
        if summed == True and rpcwise == True:
            data1 = np.sum(RPC1, axis = 1)
            data2 = np.sum(RPC2, axis = 1)
            data3 = np.sum(RPC3, axis = 1)
            data4 = np.sum(RPC4, axis = 1)
            
            filename = f'{name}_rpcwise_summed'
            
        if summed == False and rpcwise == True:
            data1 = np.ravel(RPC1)
            data2 = np.ravel(RPC2)
            data3 = np.ravel(RPC3)
            data4 = np.ravel(RPC4)
            
            filename = f'{name}_rpcwise_non_summed'
            
        if summed == True and rpcwise == True or summed == False and rpcwise == True:
            # Apply mask to remove zeroes
            mask = data1 != 0
            data1 = data1[mask]
            
            mask = data2 != 0
            data2 = data2[mask]
            
            mask = data3 != 0
            data3 = data3[mask]
            
            mask = data4 != 0
            data4 = data4[mask]
            
            # Create a single figure and axis
            plt.close()
            fig = plt.figure(figsize=v)
            ax = fig.add_subplot(1, 1, 1)
            
            # Plot histograms on the single axis
            ax.hist(data1, bins='auto', alpha=0.5, color='blue', label=f'T1, {len(data1)} events', density = normalized)
            ax.hist(data2, bins='auto', alpha=0.5, color='orange', label=f'T2, {len(data2)} events', density = normalized)
            ax.hist(data3, bins='auto', alpha=0.5, color='red', label=f'T3, {len(data3)} events', density = normalized)
            ax.hist(data4, bins='auto', alpha=0.5, color='green', label=f'T4, {len(data4)} events', density = normalized)
            ax.legend()
            ax.set_title(f'{new_title}')
            ax.set_xlim(-5, x_axes_limit_plots)
            
        if rpcwise == False and summed == False:
            Q = np.vstack((RPC1, RPC2, RPC3, RPC4))
            Q = np.ravel(Q)
            
            filename = f'{name}_total_non_summed'
            
        if rpcwise == False and summed == True:
            Q = np.vstack((RPC1, RPC2, RPC3, RPC4))
            Q = np.sum(Q, axis = 1)
            
            filename = f'{name}_total_summed'
            
        if rpcwise == False and summed == False or rpcwise == False and summed == True:
            data = Q[Q != 0]
            
            # Apply mask to remove zeroes
            mask = data != 0
            data = data[mask]
            
            # Create a single figure and axis
            plt.close()
            fig = plt.figure(figsize=v)
            ax = fig.add_subplot(1, 1, 1)
            
            # Plot histograms on the single axis
            ax.hist(data, bins='auto', alpha=0.5, color='blue', \
                    label=f'All layers, {len(data)} events', density = normalized)
            ax.legend()
            ax.set_title(f'{new_title}')
            ax.set_xlim(-5, x_axes_limit_plots)
            
    # Show the plot
    plt.xlabel('Charge (AU)')
    plt.ylabel('Counts')
    plt.tight_layout()
    name = filename
    plt.savefig(f'{output_order}_{name}.pdf', format="pdf")
    output_order = output_order + 1
    if show_plots: plt.show(); plt.close()


def multiplot_for_lip_paper(RPC1, my_title, normalized, name):
    global output_order
    
    name = f'{start}_{name}'
    new_title = f'{my_title} \nfrom {start} to {end}'
    
    filename = f'{name}_stripwise'
    
    data1_0 = RPC1[:,0]
    data1_1 = RPC1[:,1]
    data1_2 = RPC1[:,2]
    data1_3 = RPC1[:,3]
    
    # Apply mask to remove zeroes
    mask = data1_0 != 0
    data1_0 = data1_0[mask]
    mask = data1_1 != 0
    data1_1 = data1_1[mask]
    mask = data1_2 != 0
    data1_2 = data1_2[mask]
    mask = data1_3 != 0
    data1_3 = data1_3[mask]
    
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))
    
    bin_num = 'auto'
    
    # Plot histograms on each subplot
    axes[0, 0].hist(data1_0, bins=bin_num, alpha=0.5, color='blue', density = normalized)
    axes[0, 0].set_title('T1, z = 0, strip 1')
    
    # Plot histograms on other subplots using the same pattern
    axes[0, 1].hist(data1_1, bins=bin_num, alpha=0.5, color='purple', density = normalized)
    axes[0, 1].set_title('T1, z = 0, strip 2')
    
    axes[1, 0].hist(data1_2, bins=bin_num, alpha=0.5, color='red', density = normalized)
    axes[1, 0].set_title('T1, z = 0, strip 3')
    
    axes[1, 1].hist(data1_3, bins=bin_num, alpha=0.5, color='green', density = normalized)
    axes[1, 1].set_title('T1, z = 0, strip 4')
    
    for ax in axes.flatten():
        ax.set_xlabel('Charge (AU)')
        ax.set_ylabel('Counts')
        ax.set_xlim(-5, x_axes_limit_plots)  # The second argument (None) means "beyond"
    
    plt.suptitle(f'{new_title}', fontsize=16)
    plt.tight_layout()
            
    # Show the plot
    plt.xlabel('Charge (AU)')
    plt.ylabel('Counts')
    plt.tight_layout()
    name = filename
    plt.savefig(f'{output_order}_{name}.pdf', format="pdf")
    output_order = output_order + 1
    if show_plots: plt.show(); plt.close()

# -----------------------------------------------------------------------------
# Every event -----------------------------------------------------------------
# -----------------------------------------------------------------------------

multiplot(Q1_strip, Q2_strip, Q3_strip, Q4_strip, \
                    'Charge in all hits', normalized = normalized_condition,\
                        stripwise = True, summed = False, rpcwise = True, name = "total")
multiplot(Q1_strip, Q2_strip, Q3_strip, Q4_strip, \
                  'Charge in all hits', normalized = normalized_condition, \
                      stripwise = False, summed = True, rpcwise = True, name = "total")
multiplot(Q1_strip, Q2_strip, Q3_strip, Q4_strip, \
                  'Charge in all hits', normalized = normalized_condition, \
                      stripwise = False, summed = True, rpcwise = False, name = "total")

# -----------------------------------------------------------------------------
# Single ----------------------------------------------------------------------
# -----------------------------------------------------------------------------

# Stripwise charge in single hits
multiplot(Q1_ind_strip, Q2_ind_strip, Q3_ind_strip, Q4_ind_strip, \
                    'Charge in single hits', normalized = normalized_condition,\
                        stripwise = True, summed = False, rpcwise = True, name = "single")
multiplot(Q1_ind_strip, Q2_ind_strip, Q3_ind_strip, Q4_ind_strip, \
                  'Charge in single hits', normalized = normalized_condition, \
                      stripwise = False, summed = False, rpcwise = True, name = "single")
multiplot(Q1_ind_strip, Q2_ind_strip, Q3_ind_strip, Q4_ind_strip, \
                  'Charge in single hits', normalized = normalized_condition, \
                      stripwise = False, summed = False, rpcwise = False, name = "single")  

# -----------------------------------------------------------------------------
# Double ----------------------------------------------------------------------
# -----------------------------------------------------------------------------

# Preamble --------------------------------------------------------------------
Q_double_strip = np.vstack((Q1_double_strip, Q2_double_strip, Q3_double_strip, Q4_double_strip))
Q_double_non_sum = np.ravel(Q_double_strip)
Q_double_non_sum = Q_double_non_sum[Q_double_non_sum != 0]
Q_double_min = np.where(Q_double_strip != 0, Q_double_strip, np.inf)
Q_double_min = np.min(Q_double_min, axis=1)
Q_double_min = np.where(Q_double_min == np.inf, 0, Q_double_min)
Q_double_max = np.max(Q_double_strip, axis=1)
Q_double_sum = Q_double_max + Q_double_min
Q_double_sum = Q_double_sum[1:]
Q_double_max = Q_double_max[1:]
Q_double_min = Q_double_min[1:]

bin_number = 1000 # 'auto' is admited here
normalized = False

# Minimum value of charge for the double hits ---------------------------------
plt.close()
fig = plt.figure(figsize=v)

m, bins, patches = plt.hist(x=Q_double_sum, bins=bin_number, color='orange',\
                            label = 'Summed charges on both strips', alpha=0.5, density = normalized)
m, bins, patches = plt.hist(x=Q_double_min, bins=bin_number, color='blue',\
                            label = 'Minimum charge', alpha=0.6, density = normalized)
m, bins, patches = plt.hist(x=Q_double_max, bins=bin_number, color='red',\
                            label = 'Maximum charge ', alpha=0.6, density = normalized)
    
plt.title(f'Values of charge in a strip in double hits \nfrom {start} to {end}')
plt.xlabel('Charge (AU)')
plt.ylabel('Counts')
plt.xlim([-0.6, 12])
plt.legend()
# plt.xlim([-0.6, 2])
plt.tight_layout()
name = 'double_minimum'
name = f'{start}_{name}'
plt.savefig(f'{output_order}_{name}.pdf', format="pdf")
output_order = output_order + 1
if show_plots: plt.show(); plt.close()

# Maximum value of charge for the double hits ---------------------------------
plt.close()
fig = plt.figure(figsize=v)

m, bins, patches = plt.hist(x=Q_double_max, bins=1000, color='blue', alpha=0.8)
plt.title(f'Maximum value of charge in a strip in double hits \nfrom {start} to {end}')
plt.xlabel('Charge (AU)')
plt.ylabel('Counts')
plt.xlim([-0.6, 30])
plt.tight_layout()
name = 'double_maximum'
name = f'{start}_{name}'
plt.savefig(f'{output_order}_{name}.pdf', format="pdf")
output_order = output_order + 1
if show_plots: plt.show(); plt.close()

# Multiplot charge in double hits ---------------------------------------------
multiplot(Q1_double_strip, Q2_double_strip, Q3_double_strip, Q4_double_strip, \
                    'Charge per strip in double hits', normalized = normalized_condition,\
                        stripwise = True, summed = False, rpcwise = True, name = "double")
multiplot(Q1_double_strip, Q2_double_strip, Q3_double_strip, Q4_double_strip, \
                  'Charge/event in double hits', normalized = normalized_condition, \
                      stripwise = False, summed = True, rpcwise = True, name = "double")
multiplot(Q1_double_strip, Q2_double_strip, Q3_double_strip, Q4_double_strip, \
                  'Normalized charge/event in double hits', normalized = True, \
                      stripwise = False, summed = True, rpcwise = True, name = "double_norm")
multiplot(Q1_double_strip, Q2_double_strip, Q3_double_strip, Q4_double_strip, \
                  'Charge/event in double hits', normalized = normalized_condition, \
                      stripwise = False, summed = True, rpcwise = False, name = "double")

    
# -----------------------------------------------------------------------------
# Shared charge in double hits filtering the crosstalk ------------------------
# -----------------------------------------------------------------------------

normalized = False
original_matrix = Q_double_strip

new_matrices = []
# Step 4: Loop through each row of the original matrix
for row in original_matrix:
    # Step 5: Filter out the non-zero values
    non_zero_values = row[row != 0]
    
    if non_zero_values.size == 0:
        continue
    
    # Filter the crosstalk events
    if non_zero_values[0] < crosstalk_bound or non_zero_values[1] < crosstalk_bound:
        continue
    
    # Step 6: Create a new matrix with the non-zero values
    new_matrix = np.zeros_like(row)
    new_matrix[:len(non_zero_values)] = non_zero_values
    
    # Append the new matrix to the list
    new_matrices.append(new_matrix)

# Step 7: Convert the list of matrices to a NumPy array
result_matrix = np.array(new_matrices)
result_matrix = result_matrix[:, 0:2]
ratios = result_matrix[:,0]/ ( result_matrix[:,0] + result_matrix[:,1] )
ratios = np.where(ratios > 1, 1 / ratios, ratios)
ratios = ratios[ratios > 0]

# Create a single figure and axis
plt.close()
fig = plt.figure(figsize=v)
ax = fig.add_subplot(1, 1, 1)

# Plot histograms on the single axis
# bin_number_shared = 400
bin_number_shared = 'auto'

ax.hist(ratios, bins=bin_number_shared, alpha=0.6, color='blue', density = normalized)
ax.set_xlim(0, 1)
ax.set_title(f'Ratio of shared charge in double hits (no crosstalk) \nfrom {start} to {end}')

# Adjust layout
plt.xlabel('Proportion of charge shared between strips')
plt.ylabel('Counts')
plt.tight_layout()
name = 'double_shared_charge_no_crosstalk'
name = f'{start}_{name}'
plt.savefig(f'{output_order}_{name}.pdf', format="pdf")
output_order = output_order + 1
if show_plots: plt.show(); plt.close()


# Shared charge in double hits with crosstalk ---------------------------------
normalized = False

original_matrix = Q_double_strip

new_matrices = []
# Step 4: Loop through each row of the original matrix
for row in original_matrix:
    # Step 5: Filter out the non-zero values
    non_zero_values = row[row != 0]
    
    # Step 6: Create a new matrix with the non-zero values
    new_matrix = np.zeros_like(row)
    new_matrix[:len(non_zero_values)] = non_zero_values
    
    # Append the new matrix to the list
    new_matrices.append(new_matrix)


result_matrix = np.array(new_matrices)

result_matrix = result_matrix[1:, :]

ratios_min = result_matrix[:,0] / ( (result_matrix[:,0]) + (result_matrix[:,1]) )
ratios_min = np.where(ratios_min > 1, 1 / ratios_min, ratios_min)
ratios_min = ratios_min[ratios_min > 0]

ratios_max = result_matrix[:,1] / ( (result_matrix[:,0]) + (result_matrix[:,1]) )
ratios_max = np.where(ratios_max > 1, 1 / ratios_max, ratios_max)
ratios_max = ratios_max[ratios_max > 0]

# Create a single figure and axis
plt.close()
fig = plt.figure(figsize=v)
ax = fig.add_subplot(1, 1, 1)

# Plot histograms on the single axis
ax.hist(ratios_min, bins=100, alpha=0.6, color='blue', density = normalized)
ax.set_xlim(0, 1)
# ax.set_ylim(0, 150)
ax.set_title(f"Ratio of shared charge in double hits \nfrom {start} to {end}")

# Adjust layout
plt.xlabel('Proportion of charge shared between strips')
plt.ylabel('Counts')
plt.tight_layout()
name = 'double_shared_charge'
name = f'{start}_{name}'
plt.savefig(f'{output_order}_{name}.pdf', format="pdf")
output_order = output_order + 1
if show_plots: plt.show(); plt.close()

# For the LIP paper -----------------------------------------------------------
multiplot_for_lip_paper(Q1_double_strip, 'Stripwise charge spectra in double hits',\
                        False, name = "double_stripwise_RPC1")
    
# -----------------------------------------------------------------------------
# Triple ----------------------------------------------------------------------
# -----------------------------------------------------------------------------

# Preamble --------------------------------------------------------------------
Q_triple_strip = np.vstack((Q1_triple_strip, Q2_triple_strip, Q3_triple_strip, Q4_triple_strip))

# Maximum value of charge for the triple hits ---------------------------------
plt.close()
fig = plt.figure(figsize=v)

Q_triple_max = np.max(Q_triple_strip, axis=1)
m, bins, patches = plt.hist(x=Q_triple_max, bins='auto', color='blue', alpha=0.5)
plt.title(f'Maximum value of charge in a strip in triple hits \nfrom {start} to {end}')
plt.xlabel('Charge (AU)')
plt.ylabel('Counts')
plt.tight_layout()
name = 'triple_maximum'
name = f'{start}_{name}'
plt.savefig(f'{output_order}_{name}.pdf', format="pdf")
output_order = output_order + 1
if show_plots: plt.show(); plt.close()

# Minimum value of charge for the triple hits ---------------------------------
Q_triple_min = np.where(Q_triple_strip != 0, Q_triple_strip, np.inf)
Q_triple_min = np.min(Q_triple_min, axis=1)
Q_triple_min = np.where(Q_triple_min == np.inf, 0, Q_triple_min)

plt.close()
fig = plt.figure(figsize=v)

m, bins, patches = plt.hist(x=Q_triple_min, bins='auto', color='blue', alpha=0.5)
plt.title(f'Minimum value of charge in a strip in triple hits \nfrom {start} to {end}')
plt.xlabel('Charge (AU)')
plt.ylabel('Counts')
plt.xlim([-0.6, 10])
plt.tight_layout()
name = 'triple_minimum'
name = f'{start}_{name}'
plt.savefig(f'{output_order}_{name}.pdf', format="pdf")
output_order = output_order + 1
if show_plots: plt.show(); plt.close()

# Find the two lowest non-zero values for each row ----------------------------
plt.close()
fig = plt.figure(figsize=v)

Q_triple_non_zero = np.where(Q_triple_strip != 0, Q_triple_strip, np.inf)
Q_triple_sorted = np.sort(Q_triple_non_zero, axis=1)
Q_triple_two_lowest = Q_triple_sorted[:, :2]
Q_triple_two_lowest = np.where(Q_triple_two_lowest == np.inf, 0, Q_triple_two_lowest)
m, bins, patches = plt.hist(x=Q_triple_two_lowest.flatten(), bins='auto', color='blue', alpha=0.5)
plt.title(f'Two minimum Values in a Strip in Triple Hits \nfrom {start} to {end}')
plt.xlabel('Charge (AU)')
plt.ylabel('Counts')
plt.xlim([-0.6, 10])
plt.tight_layout()
name = 'triple_two_minimum'
name = f'{start}_{name}'
plt.savefig(f'{output_order}_{name}.pdf', format="pdf")
output_order = output_order + 1
if show_plots: plt.show(); plt.close()

# Stripwise charge in triple hits ---------------------------------------------
multiplot(Q1_triple_strip, Q2_triple_strip, Q3_triple_strip, Q4_triple_strip, \
                    'Charge per strip in triple hits', normalized = normalized_condition,\
                        stripwise = True, summed = False, rpcwise = True, name = "triple")
multiplot(Q1_triple_strip, Q2_triple_strip, Q3_triple_strip, Q4_triple_strip, \
                  'Charge/event in triple hits', normalized = normalized_condition, \
                      stripwise = False, summed = True, rpcwise = True, name = "triple")
multiplot(Q1_triple_strip, Q2_triple_strip, Q3_triple_strip, Q4_triple_strip, \
                  'Charge/event in triple hits', normalized = normalized_condition, \
                      stripwise = False, summed = True, rpcwise = False, name = "triple")

# -----------------------------------------------------------------------------
# Quadruple -------------------------------------------------------------------
# -----------------------------------------------------------------------------

# Preamble --------------------------------------------------------------------
Q_quad_strip = np.vstack((Q1_quad_strip, Q2_quad_strip, Q3_quad_strip, Q4_quad_strip))

# Maximum value of charge for the quad hits -----------------------------------
Q_quad_max = np.max(Q_quad_strip, axis=1)

plt.close()
fig = plt.figure(figsize=v)

m, bins, patches = plt.hist(x=Q_quad_max, bins='auto', color='blue', alpha=0.5)
plt.title(f'Maximum value of charge in a strip in quad hits \nfrom {start} to {end}')
plt.xlabel('Charge (AU)')
plt.ylabel('Counts')
plt.tight_layout()
name = 'quad_maximum'
name = f'{start}_{name}'
plt.savefig(f'{output_order}_{name}.pdf', format="pdf")
output_order = output_order + 1
if show_plots: plt.show(); plt.close()

# Minimum value of charge for the quad hits -----------------------------------
Q_quad_min = np.where(Q_quad_strip != 0, Q_quad_strip, np.inf)
Q_quad_min = np.min(Q_quad_min, axis=1)
Q_quad_min = np.where(Q_quad_min == np.inf, 0, Q_quad_min)

plt.close()
fig = plt.figure(figsize=v)

m, bins, patches = plt.hist(x=Q_quad_min, bins='auto', color='blue', alpha=0.5)
plt.title(f'Minimum value of charge in a strip in quad hits \nfrom {start} to {end}')
plt.xlabel('Charge (AU)')
plt.ylabel('Counts')
plt.xlim([-0.6, 20])
plt.tight_layout()
name = 'quad_minimum'
name = f'{start}_{name}'
plt.savefig(f'{output_order}_{name}.pdf', format="pdf")
output_order = output_order + 1
if show_plots: plt.show(); plt.close()

# Find the two lowest non-zero values for each row ----------------------------
Q_quad_non_zero = np.where(Q_quad_strip != 0, Q_quad_strip, np.inf)
Q_quad_sorted = np.sort(Q_quad_non_zero, axis=1)
Q_quad_two_lowest = Q_quad_sorted[:, :2]
Q_quad_two_low_flat = Q_quad_two_lowest.flatten()
Q_quad_two_low_flat = np.where(Q_quad_two_low_flat == np.inf, 0, Q_quad_two_low_flat)

plt.close()
fig = plt.figure(figsize=v)

m, bins, patches = plt.hist(x=Q_quad_two_low_flat, bins='auto', color='blue', alpha=0.5)
plt.title(f'Two Lowest Non-zero Values in a Strip in quad Hits \nfrom {start} to {end}')
plt.xlabel('Charge (AU)')
plt.ylabel('Counts')
plt.xlim([-0.6, 20])
plt.tight_layout()
name = 'quad_two_minimum'
name = f'{start}_{name}'
plt.savefig(f'{output_order}_{name}.pdf', format="pdf")
output_order = output_order + 1
if show_plots: plt.show(); plt.close()

# Find the three lowest non-zero values for each row --------------------------
Q_quad_strip = Q_quad_strip[1:]

Q_quad_non_zero = np.where(Q_quad_strip != 0, Q_quad_strip, np.inf)
Q_quad_sorted = np.sort(Q_quad_non_zero, axis=1)
Q_quad_three_lowest = Q_quad_sorted[:, :3]
Q_quad_three_low_flat = Q_quad_three_lowest.flatten()
Q_quad_three_low_flat = np.where(Q_quad_three_low_flat == np.inf, 0, Q_quad_three_low_flat)

plt.close()
fig = plt.figure(figsize=v)

m, bins, patches = plt.hist(x=Q_quad_three_low_flat, bins='auto', color='blue', alpha=0.5)
plt.title(f'Three Lowest Non-zero Values in a Strip in quad Hits \nfrom {start} to {end}')
plt.xlabel('Charge (AU)')
plt.ylabel('Counts')
plt.xlim([-0.6, 20])
plt.tight_layout()
name = 'quad_three_minimum'
name = f'{start}_{name}'
plt.savefig(f'{output_order}_{name}.pdf', format="pdf")
output_order = output_order + 1
if show_plots: plt.show(); plt.close()

# Stripwise charge in quad hits -----------------------------------------------
multiplot(Q1_quad_strip, Q2_quad_strip, Q3_quad_strip, Q4_quad_strip, \
                    'Charge per strip in quad hits', normalized = normalized_condition,\
                        stripwise = True, summed = False, rpcwise = True, name = "quad")
multiplot(Q1_quad_strip, Q2_quad_strip, Q3_quad_strip, Q4_quad_strip, \
                  'Charge/event in quad hits', normalized = normalized_condition, \
                      stripwise = False, summed = True, rpcwise = True, name = "quad")
multiplot(Q1_quad_strip, Q2_quad_strip, Q3_quad_strip, Q4_quad_strip, \
                  'Charge/event in quad hits', normalized = normalized_condition, \
                      stripwise = False, summed = True, rpcwise = False, name = "quad")

    
# -----------------------------------------------------------------------------
# Charge according to number of strips triggered ------------------------------
# -----------------------------------------------------------------------------

Q = np.vstack((Q1_strip, Q2_strip, Q3_strip, Q4_strip))
Q = np.sum(Q, axis = 1)
data_total = Q[Q != 0]

mask = data_total != 0
data_total = data_total[mask]

Q_ind = np.vstack((Q1_ind_strip, Q2_ind_strip, Q3_ind_strip, Q4_ind_strip))
Q_ind = np.sum(Q_ind, axis = 1)
data_ind = Q_ind[Q_ind != 0]

mask = data_ind != 0
data_ind = data_ind[mask]

Q_double = np.vstack((Q1_double_strip, Q2_double_strip, Q3_double_strip, Q4_double_strip))
Q_double = np.sum(Q_double, axis = 1)
data_double = Q_double[Q_double != 0]

mask = data_double != 0
data_double = data_double[mask]

Q_triple = np.vstack((Q1_triple_strip, Q2_triple_strip, Q3_triple_strip, Q4_triple_strip))
Q_triple = np.sum(Q_triple, axis = 1)
data_triple = Q_triple[Q_triple != 0]

mask = data_triple != 0
data_triple = data_triple[mask]

Q_quad = np.vstack((Q1_quad_strip, Q2_quad_strip, Q3_quad_strip, Q4_quad_strip))
Q_quad = np.sum(Q_quad, axis = 1)
data_quad = Q_quad[Q_quad != 0]

mask = data_quad != 0
data_quad = data_quad[mask]

# Standard scale

# Create a single figure and axis
plt.close()
fig = plt.figure(figsize=v)
ax = fig.add_subplot(1, 1, 1)

bin_number = round( len(Q) / 1000 )

my_alpha = 0.4 # Before it was 0.5

# Plot histograms on the single axis
ax.hist(data_total, bins=bin_number, alpha=my_alpha, color='blue', \
        label=f'All hits, {len(data_total)} events', density = False)
ax.hist(data_ind, bins=bin_number, alpha=my_alpha, color='orange', \
        label=f'Single hits, {len(data_ind)} events', density = False)
ax.hist(data_double, bins=bin_number, alpha=my_alpha, color='red', \
    label=f'Double hits, {len(data_double)}', density = False)
ax.hist(data_triple, bins=bin_number, alpha=my_alpha, color='green', \
        label=f'Triple hits, {len(data_triple)} events', density = False)
ax.hist(data_quad, bins=bin_number, alpha=my_alpha, color='magenta', \
        label=f'Quadruple hits, {len(data_quad)} events', density = False)
ax.legend()
ax.set_xlim(-5, x_axes_limit_plots)
ax.set_title(f"Charge/event in multistrip \nfrom {start} to {end}")
# plt.yscale("log")
plt.xlabel('Charge (AU)')
plt.ylabel('Counts')
plt.tight_layout()
name = 'multihit_together_summed'
name = f'{start}_{name}'
plt.savefig(f'{output_order}_{name}.pdf', format="pdf")
output_order = output_order + 1
if show_plots: plt.show(); plt.close()


# Log scale
plt.close()
fig = plt.figure(figsize=v)
ax = fig.add_subplot(1, 1, 1)

bin_number = round( len(Q) / 1000 )

# Plot histograms on the single axis
ax.hist(data_total, bins=bin_number, alpha=0.5, color='blue', \
        label=f'All hits, {len(data_total)} events', density = False)
ax.hist(data_ind, bins=bin_number, alpha=0.5, color='orange', \
        label=f'Single hits, {len(data_ind)} events', density = False)
ax.hist(data_double, bins=bin_number, alpha=0.5, color='red', \
    label=f'Double hits, {len(data_double)}', density = False)
ax.hist(data_triple, bins=bin_number, alpha=0.5, color='green', \
        label=f'Triple hits, {len(data_triple)} events', density = False)
ax.hist(data_quad, bins=bin_number, alpha=0.5, color='magenta', \
        label=f'Quadruple hits, {len(data_quad)} events', density = False)
ax.legend()
ax.set_xlim(-5, x_axes_limit_plots)
ax.set_title(f"Charge/event in multistrip in log scale\nfrom {start} to {end}")
plt.xlabel('Charge (AU)')
plt.ylabel('Counts')
plt.yscale("log")
plt.tight_layout()
name = 'multihit_together_summed_log'
name = f'{start}_{name}'
plt.savefig(f'{output_order}_{name}.pdf', format="pdf")
output_order = output_order + 1
if show_plots: plt.show(); plt.close()


# -----------------------------------------------------------------------------
# Multistrip plot -------------------------------------------------------------
# -----------------------------------------------------------------------------

# Pt. 1. Normalized -----------------------------------------------------------

# Compute the histograms
strip_counts1 = np.array([ len(Q1_ind_strip), len(Q1_double_strip), len(Q1_triple_strip), len(Q1_quad_strip) ])
strip_counts2 = np.array([ len(Q2_ind_strip), len(Q2_double_strip), len(Q2_triple_strip), len(Q2_quad_strip) ])
strip_counts3 = np.array([ len(Q3_ind_strip), len(Q3_double_strip), len(Q3_triple_strip), len(Q3_quad_strip) ])
strip_counts4 = np.array([ len(Q4_ind_strip), len(Q4_double_strip), len(Q4_triple_strip), len(Q4_quad_strip) ])

strip_counts1 = strip_counts1 - 0.24
strip_counts2 = strip_counts2 - 0.084
strip_counts3 = strip_counts3 + 0.084
strip_counts4 = strip_counts4 + 0.24

# Create a figure and axis
plt.close()
fig = plt.figure(figsize=v)
ax = fig.add_subplot(1, 1, 1)

# Data
x_values = [1, 2, 3, 4]

bar_width = 0.2
bar_positions1 = [x - bar_width for x in x_values]
bar_positions2 = x_values
bar_positions3 = [x + bar_width for x in x_values]
bar_positions4 = [x + 2 * bar_width for x in x_values]

# Create bar plots
ax.bar(bar_positions1, strip_counts1/sum(strip_counts1), width=bar_width, color='orange', alpha=0.5, label='RPC1')
ax.bar(bar_positions2, strip_counts2/sum(strip_counts2), width=bar_width, color='red', alpha=0.5, label='RPC2')
ax.bar(bar_positions3, strip_counts3/sum(strip_counts3), width=bar_width, color='green', alpha=0.5, label='RPC3')
ax.bar(bar_positions4, strip_counts4/sum(strip_counts4), width=bar_width, color='magenta', alpha=0.5, label='RPC4')

# Set labels and legend
ax.set_xlabel('Amount of strips triggered')
ax.set_ylabel('Counts')
ax.legend()
ax.grid(axis='y', alpha=0.5)

ax.set_xlim(0.5, 5)
ax.set_title(f"Normalized number of strips triggered in each RPC \nfrom {start} to {end}")
plt.xticks(x_values)
# plt.yscale("log")

# Show the plot
plt.tight_layout()
name = 'multistrip_rpcwise_norm'
name = f'{start}_{name}'
plt.savefig(f'{output_order}_{name}.pdf', format="pdf")
output_order = output_order + 1
if show_plots:
    plt.show()


# Pt. 2. Non-normalized -------------------------------------------------------

# Create a figure and axis
plt.close()
fig = plt.figure(figsize=v)
ax = fig.add_subplot(1, 1, 1)

# Data
x_values = [1, 2, 3, 4]

bar_width = 0.2
bar_positions1 = [x - bar_width for x in x_values]
bar_positions2 = x_values
bar_positions3 = [x + bar_width for x in x_values]
bar_positions4 = [x + 2 * bar_width for x in x_values]

# Create bar plots
ax.bar(bar_positions1, strip_counts1, width=bar_width, color='orange', alpha=0.5, label='RPC1')
ax.bar(bar_positions2, strip_counts2, width=bar_width, color='red', alpha=0.5, label='RPC2')
ax.bar(bar_positions3, strip_counts3, width=bar_width, color='green', alpha=0.5, label='RPC3')
ax.bar(bar_positions4, strip_counts4, width=bar_width, color='magenta', alpha=0.5, label='RPC4')

# Set labels and legend
ax.set_xlabel('Amount of strips triggered')
ax.set_ylabel('Counts')
ax.legend()
ax.grid(axis='y', alpha=0.5)

ax.set_xlim(0.5, 5)
ax.set_title(f"Number of strips triggered in each RPC \nfrom {start} to {end}")
plt.xticks(x_values)

# Show the plot
plt.tight_layout()
name = 'multistrip_rpcwise'
name = f'{start}_{name}'
plt.savefig(f'{output_order}_{name}.pdf', format="pdf")
output_order = output_order + 1
if show_plots:
    plt.show()


# -----------------------------------------------------------------------------
# Multiplicity plot -----------------------------------------------------------
# -----------------------------------------------------------------------------
multiplicities = multiplicities[1:]

data1 = multiplicities[:,0]
data2 = multiplicities[:,1]
data3 = multiplicities[:,2]
data4 = multiplicities[:,3]

data1 = data1[data1 != 0]
data2 = data2[data2 != 0]
data3 = data3[data3 != 0]
data4 = data4[data4 != 0]

data1 = data1 - 0.24
data2 = data2 - 0.084
data3 = data3 + 0.084
data4 = data4 + 0.24

bin_number = 30  # Selected with precision so it displays the plot properly
normalized = True  # Whether to normalize the histograms

# Compute the histograms
counts1, bins1 = np.histogram(data1, bins=bin_number, density=normalized)
counts2, bins2 = np.histogram(data2, bins=bin_number, density=normalized)
counts3, bins3 = np.histogram(data3, bins=bin_number, density=normalized)
counts4, bins4 = np.histogram(data4, bins=bin_number, density=normalized)

# Create a figure and axis
plt.close()
fig = plt.figure(figsize=v)
ax = fig.add_subplot(1, 1, 1)

# Plot the histogram bars without lines
ax.bar(bins1[:-1], counts1/sum(counts1), width=(bins1[1] - bins1[0]), color='orange', alpha=0.5, label=f'RPC1 {len(data1)} events')
ax.bar(bins2[:-1], counts2/sum(counts2), width=(bins2[1] - bins2[0]), color='red', alpha=0.5, label=f'RPC2 {len(data2)} events')
ax.bar(bins3[:-1], counts3/sum(counts3), width=(bins3[1] - bins3[0]), color='green', alpha=0.5, label=f'RPC3 {len(data3)} events')
ax.bar(bins4[:-1], counts4/sum(counts4), width=(bins4[1] - bins4[0]), color='magenta', alpha=0.5, label=f'RPC4 {len(data4)} events')

# Set labels and legend
ax.set_xlabel('Multiplicity')
ax.set_ylabel('Counts')
ax.legend()
ax.grid(axis='y', alpha=0.5)

ax.set_xlim(0.5, 5)
ax.set_title(f"Normalized multiplicity distribution in each RPC \nfrom {start} to {end}")
plt.xticks([1, 2, 3, 4, 5])
plt.yscale("log")

# Show the plot
plt.tight_layout()
name = 'multiplicity_rpcwise_norm'
name = f'{start}_{name}'
plt.savefig(f'{output_order}_{name}.pdf', format="pdf")
output_order = output_order + 1
if show_plots: plt.show(); plt.close()

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Work in progress ------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Interstrip vs.double strip hits ---------------------------------------------
# -----------------------------------------------------------------------------

Q_n1 = np.vstack((Q1_n1_strip, Q2_n1_strip, Q3_n1_strip, Q4_n1_strip))
Q_interstrip = np.vstack((Q1_interstrip, Q2_interstrip, Q3_interstrip, Q4_interstrip))

# -----------------------------------------------------------------------------
# Induction section -----------------------------------------------------------
# -----------------------------------------------------------------------------

new_matrix = np.zeros(4)

# Loop through each row in the original matrix
for row in Q_n1:
    # Count the number of zeroes in the current row
    num_zeroes = np.count_nonzero(row)
    
    # Check if the row has two or more zeroes
    if num_zeroes >= 2:
        # If it does, add the row to the new matrix
        new_matrix = np.vstack((new_matrix, row))

Q_n1_sin_double = new_matrix

# interstrip_count = len(Q_ind) + len(Q_double) - len(Q_n1_sin_double)

# ratio = len(Q_ind) / ( len(Q_ind) + interstrip_count )
# print('Ratio of muon hits that are in only one strip: ',  round(100*ratio)/100)

# # We ponderate the assymetric strips
# print('Estimated induction section: ', \
#       round(np.pi * ( (1 - ratio)/2 *(98*0.25 + 63*0.75) )**2)/100, 'cm**2' )

# -----------------------------------------------------------------------------
# End of work in progress -----------------------------------------------------
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Position map-----------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

# I HAVE TO CORRECT THE WEIGHTS FOR THE CASE MID_POINTS IS TRUE, SINCE THE BIN
# WIDTH IS DIFFERENT, SO THE RATIO IS DIFFERENT ALSO.

weighted = False
blurred = True; blur = 1
mid_points = False
eff_corrected = True

efficient_positions = False
non_efficient_positions = False

all_positions = True

if efficient_positions:
    map_positions = positions_detected
if non_efficient_positions:
    map_positions = positions_non_detected
if all_positions:
    map_positions = positions

total_time_min = time_difference / 60

third_column_values = [0, 100, 200, 400]

# Define non-uniform bin edges
x_bins = np.linspace(0, 300, 31)

if mid_points:
    # Actually this bins should be written with the spacing between point so
    # the width of the mid bands is just the diameter of the ionizing/induction
    # section; so the weighting will actually be uniform.
    
    y_bins_1 = [0, 76.0, 112.0, 143.5, 174.5, 210.875, 239.75, 287]
    y_bins_2 = [0, 35.875, 76.0, 112.0, 143.5, 174.5, 210.875, 287]
else:
    y_bins_1 = [0, 98, 161, 224, 287]
    y_bins_2 = [0, 63, 126, 189, 287]

fig, axes = plt.subplots(2, 2, figsize=(10,8))

if eff_corrected:
    fig.suptitle(f'Corrected by efficiency hit position distribution \nfrom {start} to {end}')
else:
    fig.suptitle(f'Hit position distribution \nfrom {start} to {end}')

# Loop through the different third column values
for idx, third_value in enumerate(third_column_values):
    filtered_positions = map_positions[map_positions[:, 2] == third_value]
    
    x = filtered_positions[:, 0]
    y = filtered_positions[:, 1]
    
    z = third_value
    
    row = idx // 2
    col = idx % 2
    ax = axes[row, col]
    
    # Write weights that transform to counts / cm / min (not cm² but cm/100)
    weights = np.ones_like(y) / (300) * 10000  / total_time_min
    
    # Create a 2D histogram-like colormap with non-uniform bins
    if z == 0 or z == 200:
        y_bins = y_bins_1
        
        if mid_points:
            for i in range(len(y_bins) - 1):
                # Use '&' for element-wise 'and'
                conditions = (y > y_bins[i]) & (y < y_bins[i + 1])
                distances = np.diff(y_bins)[i]
                weights = np.where(conditions, weights / distances, weights)
        else:
            weights = np.where(y < 98, weights / 98, weights / 63)
                
        if z == 0:
            # weights = weights / 3.2 # Stereoradians
            if eff_corrected:
                weights = weights / eff1
        if z == 200:
            # weights = weights / 3.2 # Stereoradians
            if eff_corrected:
                weights = weights / eff3

                    
    if z == 100 or z == 400:
        y_bins = y_bins_2
        
        if mid_points:
            for i in range(len(y_bins) - 1):
                # Use '&' for element-wise 'and'
                conditions = (y > y_bins[i]) & (y < y_bins[i + 1])
                distances = np.diff(y_bins)[i]
                weights = np.where(conditions, weights / distances, weights)
        else:
            weights = np.where(y > 168, weights / 98, weights / 63)
        
        if z == 100:
            # weights = weights / 3.2 # Stereoradians
            if eff_corrected:
                weights = weights / eff2
        if z == 400:
            # weights = weights / 1.615 # Stereoradians
            if eff_corrected:
                weights = weights / eff4

    if weighted == False:
        weights = np.ones_like(y)
    
    # Calculate the histogram
    H, _, _ = np.histogram2d(x, y, bins=[x_bins, y_bins], weights=weights)
    pcm = ax.pcolormesh(x_bins, y_bins, H.T, cmap='viridis', shading='auto')
    
    # Invert the x-axis for reverse direction
    ax.invert_xaxis()
    
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    
    if z == 0: RPC = 1
    if z == 100: RPC = 2
    if z == 200: RPC = 3
    if z == 400: RPC = 4
    
    ax.set_title(f'T{RPC}, z = {z}, {len(x)} counts')
    ax.set_aspect('equal')
    
    if eff_corrected:
        if z == 0:
            flux = len(x)/ (300*300) * 10000  / total_time_min / eff1
        if z == 100:
            flux = len(x)/ (300*300) * 10000  / total_time_min / eff2
        if z == 200:
            flux = len(x)/ (300*300) * 10000  / total_time_min / eff3
        if z == 400:
            flux = len(x)/ (300*300) * 10000  / total_time_min / eff4
    else:
        flux = len(x)/ (300*300) * 10000  / total_time_min
    
    flux = round(flux * 10) / 10
    
    # Add a colorbar
    cbar = plt.colorbar(pcm, ax=ax)
    if weighted:
        cbar.set_label('Count flux (counts / cm² / min)')
        ax.set_title(f'T{RPC}, z = {z}, {len(x)} counts, {flux} counts/cm²/min')
    else:
        cbar.set_label('Counts')
    
    if blurred:
        blurred_pcm = gaussian_filter(pcm.get_array(), sigma = blur)  # Adjust sigma for blurring
        pcm.set_array(blurred_pcm)

plt.tight_layout()
name = f"{start}_position_map"
plt.savefig(f'{output_order}_{name}.pdf', format="pdf")
output_order = output_order + 1
if show_plots: plt.show(); plt.close()


# # To represent the positions in Y selected (to see how much interstrip
# # was assigned)
# pos = positions
# # pos = positions[positions[:,2] == 400]
# pos = pos[:,1]
# m, bins, patches = plt.hist(x=pos, bins='auto', color='blue', alpha=0.8)
# m, bins, patches = plt.hist(x=pos, bins=[0, 35.875, 76.0, 112.0,\
#                 143.5, 174.5, 210.875, 239.75, 287], color='blue', alpha=0.8)
# plt.title(f'Charge collected per strip in double hits \nfrom {start} to {end}')
# plt.xlim([0, 287])
# if show_plots: plt.show(); plt.close()


# -----------------------------------------------------------------------------
# Rates -----------------------------------------------------------------------
# -----------------------------------------------------------------------------

# Initialize empty lists to store data_filtered
dates = []
data_filtered = []
data_raw = []
data_strongly_filtered = []

# File path
file_path = f"rates_from_{init_time}_to_{end}.txt"  # Replace 'your_file.csv' with the actual file path

# Read the file and extract data_filtered
with open(file_path, 'r') as file:
    next(file)  # Skip the first line
    
    for line in file:
        if "Rates" in line:  # Check if the line contains the word "Rates"
            continue  # Skip this line and move to the next line
            
        columns = line.strip().split()  # Assuming space-separated columns
        
        # Assuming the second and third columns are in the format 'YYYY-MM-DD'
        # date_str = columns[1:3]  # Second column
        # date_str = " ".join(date_str)
        # date = datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
        
        new_time_components = columns[1:3]
        date_components = new_time_components[0].split('-')
        time_components = new_time_components[1].split(':')
        
        datetime_str = f"{date_components[0]}-{date_components[1]}-{date_components[2]} {time_components[0]}:{time_components[1]}:{time_components[2]}"
        
        date = pd.to_datetime(datetime_str, format="%Y-%m-%d %H:%M:%S")
        
        # Seventh column as data_filtered (assuming it's numeric)
        datum_filtered = float(columns[6])  # Seventh column
        datum_raw = float(columns[7])
        datum_strongly_filtered = float(columns[8])
        
        dates.append(date)
        
        data_filtered.append(datum_filtered)
        data_raw.append(datum_raw)
        data_strongly_filtered.append(datum_strongly_filtered)
    

# Convert lists to numpy arrays for plotting
dates = np.array(dates)

# data_filtered = np.array(data_filtered)
# data_raw = np.array(data_raw)
# data_strongly_filtered = np.array(data_strongly_filtered)

if len(dates) == 0:
    print('Not enough data_filtered to create a rate. Resuming execution...')
else:
    # Create a scatter plot
    plt.close()
    plt.figure(figsize=v)
    plt.scatter(dates, data_filtered, marker='o', color='b', alpha = 0.7, label = 'Filtered data')
    # plt.scatter(dates, data_raw, marker='o', color='r', alpha = 0.7, label = 'Raw data')
    plt.scatter(dates, data_strongly_filtered, marker='o', color='g', alpha = 0.7, label = 'Strongly filtered data')
    
    x_min = min(dates)
    x_max = max(dates)
    
    # Set the x-limits to be tight to your data_filtered
    plt.xlim(x_min, x_max)
    
    # Customize the plot
    plt.xlabel('Date')
    plt.ylabel('Rate (cts/hr)')
    plt.grid(True)
    plt.legend()
    
    # Rotate x-axis labels for better readability (optional)
    plt.xticks(rotation=45)
    
    # Show the plot
    plt.title(f'Rate vs. Date from {x_min} to {x_max}')
    plt.tight_layout()
    name = f"{start}_rates"
    plt.savefig(f'{output_order}_{name}.pdf', format="pdf")
    output_order = output_order + 1
    if show_plots: plt.show(); plt.close()

if len(dates) == 0:
    print('Not enough data_filtered to create a rate. Resuming execution...')
else:
    # Create a scatter plot
    plt.close()
    plt.figure(figsize=v)
    # plt.scatter(dates, data_filtered, marker='o', color='b', alpha = 0.7, label = 'Filtered data')
    plt.scatter(dates, data_raw, marker='o', color='r', alpha = 0.7, label = 'Raw data')
    # plt.scatter(dates, data_strongly_filtered, marker='o', color='g', alpha = 0.7, label = 'Strongly filtered data')
    
    x_min = min(dates)
    x_max = max(dates)
    
    # Set the x-limits to be tight to your data_filtered
    plt.xlim(x_min, x_max)
    
    # Customize the plot
    plt.xlabel('Date')
    plt.ylabel('Rate (cts/hr)')
    plt.grid(True)
    plt.legend()
    
    # Rotate x-axis labels for better readability (optional)
    plt.xticks(rotation=45)
    
    # Show the plot
    plt.title(f'Raw rate vs. Date from {x_min} to {x_max}')
    plt.tight_layout()
    name = f"{start}_rates_raw"
    plt.savefig(f'{output_order}_{name}.pdf', format="pdf")
    output_order = output_order + 1
    if show_plots: plt.show(); plt.close()

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Report creation -------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

x = [a for a in os.listdir() if a.endswith(".pdf")]

merger = PdfMerger()

y = sorted(x, key=lambda s: int(s.split('_')[0]))

for pdf in y:
    merger.append(open(pdf, 'rb'))

with open(f"report_daq_data_from_{start}_to_{end}.pdf", "wb") as fout:
    merger.write(fout)

print('----------------------------------------------------------------------')
print(f"Report stored as 'report_{start}_to_{end}.pdf'")

# Execution time
end_time = time.time()
exec_time_s = end_time - init_exec_time

print('----------------------------------------------------------------------')
print('Analysis finished')
print(f'Took {round(exec_time_s/60*10)/10} min = {round(exec_time_s/60/60*10)/10} h')
print('----------------------------------------------------------------------')


os.chdir("../../../analysis_scripts")
