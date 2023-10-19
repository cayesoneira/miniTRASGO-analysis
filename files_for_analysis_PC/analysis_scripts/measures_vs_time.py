#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 11:58:02 2023

@author: cayesoneira
"""

# -----------------------------------------------------------------------------
# Header ----------------------------------------------------------------
# -----------------------------------------------------------------------------

from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from PyPDF2 import PdfMerger

# -----------------------------------------------------------------------------
# REPRESENTING ----------------------------------------------------------------
# -----------------------------------------------------------------------------
#   1. TEMPERATURE
temp_min_C = 10
temp_max_C = 30

#   2. PRESSURE
#   3. RH
#   4. GAS FLOW
flow_min_AU = 470
flow_max_AU = 1050

#   5. VOLTAGE
voltage_min_kV = 5
voltage_max_kV = 6

#   6. TRB RATES
trb_trigger_min_cts_s = 0
trb_trigger_max_cts_s = 15

trb_coincidence_min_cts_s = 0
trb_coincidence_max_cts_s = 10

#   7. FILTERED RATES
filtered_rate_min_cts_hr = 20000
filtered_rate_max_cts_hr = 27000

#   8. CHARGE STUFF
charge_min_AU = -5
charge_max_AU = 120

#   9. EFFICIENCIES
#   10. MEAN MULTIPLICITIES
mean_multiplicity_min = 1
mean_multiplicity_max = 1.08

# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# Some important variables ----------------------------------------------------
# -----------------------------------------------------------------------------

print('--------------------------------------------------------------')
try:
    sys.argv[1]
except IndexError:
    print("No input date given. Using all the data stored...")
    
    # Get the current date
    end_date = datetime.now()
    
    # Calculate the last day of the current month
    start_date = datetime(end_date.year, end_date.month, end_date.day) - timedelta(days=100)
else:
    print("Running with input given in an external script.")
    # Get the start and end date as command line arguments
    start_date_str = sys.argv[1]
    end_date_str = sys.argv[2]
    
    # Parse the input strings into datetime objects
    try:
        # Assuming the input format is YYMMDD
        
        # Parse start date
        start_year = int(start_date_str[:2])
        start_month = int(start_date_str[2:4])
        start_day = int(start_date_str[4:6])
        
        # Parse end date
        end_year = int(end_date_str[:2])
        end_month = int(end_date_str[2:4])
        end_day = int(end_date_str[4:6])
        
        start_year = int(f"20{start_year}")
        end_year = int(f"20{end_year}")
        
        # Create datetime objects
        start_date = datetime(year=start_year, month=start_month, day=start_day)
        end_date = datetime(year=end_year, month=end_month, day=end_day)
        
    except ValueError as e:
        print("Error:", e)
        sys.exit(1)


# With time interval
analysis_duration_in_h = 24*1 # In hours

show_plots = False

# Markersize
size = 5

v=(14,4)

# -----------------------------------------------------------------------------
# Create the directory to work if it does not exist and move to it ------------
# -----------------------------------------------------------------------------

directory_name = "Results_vs_Time"

# Specify the path where you want to create the directory
base_directory = "../Data_and_Results/"

# Combine the base directory and the new directory name
new_directory_path = os.path.join(base_directory, directory_name)

# Check if the directory already exists

print('--------------------------------------------------------------')
if not os.path.exists(new_directory_path):
    # Create the directory
    os.mkdir(new_directory_path)
    print(f"Created directory: {new_directory_path}")
else:
    print(f"Directory already exists: {new_directory_path}")
print('--------------------------------------------------------------')

# And cd to it so the results are stored there
os.chdir(new_directory_path)


# -----------------------------------------------------------------------------
# Move to a subdirectory ------------------------------------------------------
# -----------------------------------------------------------------------------

directory_name = f"Results_vs_Time_from_{start_date}_to_{end_date}"

# Specify the path where you want to create the directory
base_directory = "."

# Combine the base directory and the new directory name
new_directory_path = os.path.join(base_directory, directory_name)

# Check if the directory already exists

print('--------------------------------------------------------------')
if not os.path.exists(new_directory_path):
    # Create the directory
    os.mkdir(new_directory_path)
    print(f"Created directory: {new_directory_path}")
else:
    print(f"Directory already exists: {new_directory_path}")
print('--------------------------------------------------------------')

# And cd to it so the results are stored there
os.chdir(new_directory_path)


# -----------------------------------------------------------------------------
# A function to import dates --------------------------------------------------
# -----------------------------------------------------------------------------

def import_date(date_str):
    try:
        date_parts = date_str.split()
        if len(date_parts) != 2:
            return None  # Not in the expected format

        date_components = date_parts[0].split('-') + date_parts[1].split(':')
        if len(date_components) != 6:
            return None  # Not in the expected format

        year, month, day, hour, minute, second = map(int, date_components)

        date = datetime(year, month, day, hour, minute, second)
    except ValueError:
        return None
    else:
        return date


# -----------------------------------------------------------------------------
# 1. Temperature --------------------------------------------------------------
# -----------------------------------------------------------------------------

# Internal temperatures -------------------------------------------------------

current_directory = os.getcwd()
print("Current Directory:", current_directory)

file_path_data = '../../Log_data/merged_sensors_bus0.txt'

dates_log = []
data_temp = []

with open(file_path_data, 'r') as file:
    next(file)  # Skip the first line
    
    for line in file:
        if "Rates" in line or "----" in line:
            continue  # Skip this line and move to the next line

        columns = line.strip().split()  # Assuming space-separated columns
        
        if columns[0] == 'nan' or columns[1] == 'nan':
            continue
        
        # Assuming the second and third columns are in the format 'YYYY-MM-DD'
        date_str = columns[0:2]  # Second column
        date_str = " ".join(date_str)
        
        date = import_date(date_str)
        if date is None:
            continue
        
        if start_date <= date <= end_date:
            # Seventh column as data_temp (assuming it's numeric)
            try:
                float(columns[2])
            except IndexError:
                continue
            else:
                datum = float(columns[2])  # Seventh column
        
            dates_log.append(date)
            data_temp.append(datum)

# Convert lists to numpy arrays for plotting
dates_log = np.array(dates_log)
data_temp = np.array(data_temp)

dates_temp = dates_log
temp_ext = data_temp

x_min_temp = min(dates_log)
x_max_temp = max(dates_log)

# Create a scatter plot
plt.figure(figsize=v)
plt.scatter(dates_log, data_temp, marker=',', s = size, color='b', alpha = 0.7, label = 'External temperature')


# External temperatures -------------------------------------------------------

dates_log = []
data_temp = []

file_path_data = '../../Log_data/merged_sensors_bus1.txt'

with open(file_path_data, 'r') as file:
    next(file)  # Skip the first line
    
    for line in file:
        if "rates" in line:  # Check if the line contains the word "Rates"
            continue  # Skip this line and move to the next line
            
        columns = line.strip().split()  # Assuming space-separated columns
        
        if columns[0] == 'nan' or columns[1] == 'nan':
            continue
        
        # Assuming the second and third columns are in the format 'YYYY-MM-DD'
        date_str = columns[0:2]  # Second column
        date_str = " ".join(date_str)
        
        date = import_date(date_str)
        if date is None:
            continue
        
        if start_date <= date <= end_date:
            # Seventh column as data_temp (assuming it's numeric)
            try:
                float(columns[2])
            except IndexError:
                continue
            else:
                datum = float(columns[2])  # Seventh column
        
            dates_log.append(date)
            data_temp.append(datum)


# Convert lists to numpy arrays for plotting
dates_log = np.array(dates_log)
data_temp = np.array(data_temp)

temp_int = data_temp

plt.scatter(dates_log, data_temp, marker=',', s = size, color='r', alpha = 0.7, label = 'Internal temperature')

# Set the x-limits to be tight to your data_temp
plt.xlim(x_min_temp, x_max_temp)
plt.ylim(temp_min_C,temp_max_C)

# Customize the plot
plt.xlabel('Date (YYYY-MM-DD)')
plt.ylabel('Temperature (ºC)')
plt.title(f'Temp vs. Date from\n{x_min_temp} to {x_max_temp}')
plt.grid(True)

plt.legend()

# Rotate x-axis labels for better readability (optional)
plt.xticks(rotation=45)

# Show the plot
plt.tight_layout()
plt.savefig(f'temp_from_{x_min_temp}_to_{x_max_temp}.pdf', format="pdf")
if show_plots: plt.show(); plt.close()


# -----------------------------------------------------------------------------
# 2. Pressure -----------------------------------------------------------------
# -----------------------------------------------------------------------------

dates_log = []
data_temp = []

file_path_data = '../../Log_data/merged_sensors_bus0.txt'

with open(file_path_data, 'r') as file:
    next(file)  # Skip the first line
    
    for line in file:
        if "rates" in line:  # Check if the line contains the word "Rates"
            continue  # Skip this line and move to the next line
        
        columns = line.strip().split()  # Assuming space-separated columns
        
        if columns[0] == 'nan' or columns[1] == 'nan':
            continue
        
        # Assuming the second and third columns are in the format 'YYYY-MM-DD'
        date_str = columns[0:2]  # Second column
        date_str = " ".join(date_str)
        
        date = import_date(date_str)
        if date is None:
            continue
        
        if start_date <= date <= end_date:
            # Seventh column as data_temp (assuming it's numeric)
            try:
                float(columns[4])
            except IndexError:
                continue
            else:
                datum = float(columns[4])  # Seventh column
        
            dates_log.append(date)
            data_temp.append(datum)


# Convert lists to numpy arrays for plotting
dates_log = np.array(dates_log)
data_temp = np.array(data_temp)

# Create a scatter plot
plt.figure(figsize=v)
plt.scatter(dates_log, data_temp, marker=',', s = size, color='b', alpha = 0.7, label = 'External pressure')

# -----------------------------------------------------------------------------
dates_log = []
data_temp = []

file_path_data = '../../Log_data/merged_sensors_bus1.txt'

with open(file_path_data, 'r') as file:
    next(file)  # Skip the first line
    
    for line in file:
        if "rates" in line:  # Check if the line contains the word "Rates"
            continue  # Skip this line and move to the next line
            
        columns = line.strip().split()  # Assuming space-separated columns
        
        if columns[0] == 'nan' or columns[1] == 'nan':
            continue
        
        # Assuming the second and third columns are in the format 'YYYY-MM-DD'
        date_str = columns[0:2]  # Second column
        date_str = " ".join(date_str)
        
        date = import_date(date_str)
        if date is None:
            continue
        
        if start_date <= date <= end_date:
            # Seventh column as data_temp (assuming it's numeric)
            try:
                float(columns[4])
            except IndexError:
                continue
            else:
                datum = float(columns[4])  # Seventh column
        
            dates_log.append(date)
            data_temp.append(datum)


# Convert lists to numpy arrays for plotting
dates_log = np.array(dates_log)
data_temp = np.array(data_temp)

plt.scatter(dates_log, data_temp, marker=',', s = size, color='r', alpha = 0.7, label = 'Internal pressure')

x_min_temp = min(dates_log)
x_max_temp = max(dates_log)

# Set the x-limits to be tight to your data_temp
plt.xlim(x_min_temp, x_max_temp)

# Customize the plot
plt.xlabel('Date (YYYY-MM-DD)')
plt.ylabel('Pressure (mbar)')
plt.title(f'Pressure vs. Date from\n{x_min_temp} to {x_max_temp}')
plt.grid(True)

plt.legend()

# Rotate x-axis labels for better readability (optional)
plt.xticks(rotation=45)

# Show the plot
plt.tight_layout()
plt.savefig(f'pressure_from_{x_min_temp}_to_{x_max_temp}.pdf', format="pdf")
if show_plots: plt.show(); plt.close()


# -----------------------------------------------------------------------------
# 3. Humidity -----------------------------------------------------------------
# -----------------------------------------------------------------------------
dates_log = []
data_temp = []

file_path_data = '../../Log_data/merged_sensors_bus0.txt'

with open(file_path_data, 'r') as file:
    next(file)  # Skip the first line
    
    for line in file:
        if "rates" in line:  # Check if the line contains the word "Rates"
            continue  # Skip this line and move to the next line
        
        columns = line.strip().split()  # Assuming space-separated columns
        
        if columns[0] == 'nan' or columns[1] == 'nan':
            continue
        
        # Assuming the second and third columns are in the format 'YYYY-MM-DD'
        date_str = columns[0:2]  # Second column
        date_str = " ".join(date_str)
        
        date = import_date(date_str)
        if date is None:
            continue
        
        if start_date <= date <= end_date:
            # Fourth column as data_temp (assuming it's numeric)
            try:
                float(columns[3])
            except IndexError:
                continue
            else:
                datum = float(columns[3])  # Fourth column
        
            dates_log.append(date)
            data_temp.append(datum)


# Convert lists to numpy arrays for plotting
dates_log = np.array(dates_log)
data_temp = np.array(data_temp)

# Create a scatter plot
plt.figure(figsize=v)
plt.scatter(dates_log, data_temp, marker=',', s = size, color='b', alpha = 0.7, label = 'External humidity')


# -----------------------------------------------------------------------------
dates_log = []
data_temp = []

file_path_data = '../../Log_data/merged_sensors_bus1.txt'

with open(file_path_data, 'r') as file:
    next(file)  # Skip the first line
    
    for line in file:
        if "rates" in line:  # Check if the line contains the word "Rates"
            continue  # Skip this line and move to the next line
            
        columns = line.strip().split()  # Assuming space-separated columns
        
        if columns[0] == 'nan' or columns[1] == 'nan':
            continue
        
        # Assuming the second and third columns are in the format 'YYYY-MM-DD'
        date_str = columns[0:2]  # Second column
        date_str = " ".join(date_str)
        
        date = import_date(date_str)
        if date is None:
            continue
        
        if start_date <= date <= end_date:
            # Fourth column as data_temp (assuming it's numeric)
            try:
                float(columns[3])
            except IndexError:
                continue
            else:
                datum = float(columns[3])  # Fourth column
        
            dates_log.append(date)
            data_temp.append(datum)


# Convert lists to numpy arrays for plotting
dates_log = np.array(dates_log)
data_temp = np.array(data_temp)

plt.scatter(dates_log, data_temp, marker=',', s = size, color='r', alpha = 0.7, label = 'Internal humidity')

x_min_temp = min(dates_log)
x_max_temp = max(dates_log)

# Set the x-limits to be tight to your data_temp
plt.xlim(x_min_temp, x_max_temp)

# Customize the plot
plt.xlabel('Date (YYYY-MM-DD)')
plt.ylabel('RH (%)')
plt.title(f'Relative humidity vs. Date from\n{x_min_temp} to {x_max_temp}')
plt.grid(True)

plt.legend()

# Rotate x-axis labels for better readability (optional)
plt.xticks(rotation=45)

# Show the plot
plt.tight_layout()
plt.savefig(f'humidity_from_{x_min_temp}_to_{x_max_temp}.pdf', format="pdf")
if show_plots: plt.show(); plt.close()


# -----------------------------------------------------------------------------
# 4. Gas flow --------------------------------------------------------------------
# -----------------------------------------------------------------------------

dates_log = []
data_1 = []
data_2 = []
data_3 = []
data_4 = []

file_path_data = '../../Log_data/merged_Flow0.txt'

with open(file_path_data, 'r') as file:
    next(file)  # Skip the first line
    
    for line in file:
        if "rates" in line:  # Check if the line contains the word "Rates"
            continue  # Skip this line and move to the next line
            
        columns = line.strip().split()  # Assuming space-separated columns
        
        if columns[0] == 'nan' or columns[1] == 'nan':
            continue
        
        # Assuming the second and third columns are in the format 'YYYY-MM-DD'
        date_str = columns[0:2]  # Second column
        date_str = " ".join(date_str)
        
        date = import_date(date_str)
        if date is None:
            continue
        
        if start_date <= date <= end_date:
            # Assuming columns[2] to columns[5] are numeric data columns
            try:
                datum_1 = float(columns[2])
                datum_2 = float(columns[3])
                datum_3 = float(columns[4])
                datum_4 = float(columns[5])
            except IndexError:
                continue
        
            dates_log.append(date)
            data_1.append(datum_1)
            data_2.append(datum_2)
            data_3.append(datum_3)
            data_4.append(datum_4)


# Convert lists to numpy arrays for plotting
dates_log = np.array(dates_log)
data_1 = np.array(data_1)
data_2 = np.array(data_2)
data_3 = np.array(data_3)
data_4 = np.array(data_4)

# Create a scatter plot
plt.figure(figsize=v)

plt.scatter(dates_log, data_1, marker=',', s = size, color='r', alpha = 0.7, label = 'RPC 1')
plt.scatter(dates_log, data_2, marker=',', s = size, color='g', alpha = 0.7, label = 'RPC 2')
plt.scatter(dates_log, data_3, marker=',', s = size, color='b', alpha = 0.7, label = 'RPC 3')
plt.scatter(dates_log, data_4, marker=',', s = size, color='orange', alpha = 0.7, label = 'RPC 4')

x_min_temp = min(dates_log)
x_max_temp = max(dates_log)

# Set the x-limits to be tight to your data_temp
plt.xlim(x_min_temp, x_max_temp)
plt.ylim(flow_min_AU, flow_max_AU)

# Customize the plot
plt.xlabel('Date (YYYY-MM-DD)')
plt.ylabel('Gas flow (AU)')
plt.title(f'Gas flow vs. Date from\n{x_min_temp} to {x_max_temp}')
plt.grid(True)

plt.xticks(rotation=45)
plt.legend()

# Rotate x-axis labels for better readability (optional)

# Show the plot
plt.tight_layout()
plt.savefig(f'flow_from_{x_min_temp}_to_{x_max_temp}.pdf', format="pdf")
if show_plots: plt.show(); plt.close()

# -----------------------------------------------------------------------------
# 5. Voltage ------------------------------------------------------------------
# -----------------------------------------------------------------------------

dates_log = []
data_temp = []

file_path_data = '../../Log_data/merged_hv0.txt'

with open(file_path_data, 'r') as file:
    next(file)  # Skip the first line
    
    for line in file:
        if "rates" in line:  # Check if the line contains the word "Rates"
            continue  # Skip this line and move to the next line
            
        columns = line.strip().split()  # Assuming space-separated columns
        
        if columns[0] == 'nan' or columns[1] == 'nan':
            continue
        
        # Assuming the second and third columns are in the format 'YYYY-MM-DD'
        date_str = columns[0:2]  # Second column
        date_str = " ".join(date_str)
        
        date = import_date(date_str)
        if date is None:
            continue
        
        if start_date <= date <= end_date:
            # Eleventh column as data_temp (assuming it's numeric)
            try:
                float(columns[10])
            except IndexError:
                continue
            else:
                datum = float(columns[10])  # Eleventh column
        
            dates_log.append(date)
            data_temp.append(datum)


# Convert lists to numpy arrays for plotting
dates_log = np.array(dates_log)
data_temp = np.array(data_temp)

# Create a scatter plot
plt.figure(figsize=v)
plt.scatter(dates_log, data_temp, marker=',', s = size, color='b', alpha = 0.7)

x_min_temp = min(dates_log)
x_max_temp = max(dates_log)

# Set the x-limits to be tight to your data_temp
plt.xlim(x_min_temp, x_max_temp)
plt.ylim(voltage_min_kV, voltage_max_kV)

# Customize the plot
plt.xlabel('Date (YYYY-MM-DD)')
plt.ylabel('Voltage (kV)')
plt.title(f'Voltage vs. Date from\n{x_min_temp} to {x_max_temp}')
plt.grid(True)

# Rotate x-axis labels for better readability (optional)
plt.xticks(rotation=45)

# Show the plot
plt.tight_layout()
plt.savefig(f'voltage_from_{x_min_temp}_to_{x_max_temp}.pdf', format="pdf")
if show_plots: plt.show(); plt.close()


# -----------------------------------------------------------------------------
# 6. TRB trigger rates --------------------------------------------------------
# -----------------------------------------------------------------------------

# <YYYY-MM-DD> <HH:mm:ss> <Trigger asserted> <Trigger rising edges> <Trigger accepted> <Trigger multiplexer (TM) 0> <TM 1> <TM 2> <TM 3> <Coincidence Module (CM) 0> <CM 1> <CM 2> <CM 3>

dates_log = []
data_1 = []
data_2 = []
data_3 = []
data_4 = []
data_5 = []
data_6 = []
data_7 = []

file_path_data = '../../Log_data/merged_rates.txt'

with open(file_path_data, 'r') as file:
    next(file)  # Skip the first line
    
    for line in file:
        if "rates" in line:  # Check if the line contains the word "Rates"
            continue  # Skip this line and move to the next line
            
        columns = line.strip().split()  # Assuming space-separated columns
        
        if columns[0] == 'nan' or columns[1] == 'nan':
            continue
        
        # Assuming the second and third columns are in the format 'YYYY-MM-DD'
        date_str = columns[0:2]  # Second column
        date_str = " ".join(date_str)
        
        date = import_date(date_str)
        if date is None:
            continue
        
        if start_date <= date <= end_date:
            # Additional columns as data_temp (assuming they're numeric)
            try:
                datum_1 = float(columns[2])
                datum_2 = float(columns[3])
                datum_3 = float(columns[4])
                datum_4 = float(columns[9])
                datum_5 = float(columns[10])
                datum_6 = float(columns[11])
                datum_7 = float(columns[12])
            except IndexError:
                continue
        
            dates_log.append(date)
            data_1.append(datum_1)
            data_2.append(datum_2)
            data_3.append(datum_3)
            data_4.append(datum_4)
            data_5.append(datum_5)
            data_6.append(datum_6)
            data_7.append(datum_7)


# Convert lists to numpy arrays for plotting
dates_log = np.array(dates_log)
data_1 = np.array(data_1)
data_2 = np.array(data_2)
data_3 = np.array(data_3)
data_4 = np.array(data_4)
data_5 = np.array(data_5)
data_6 = np.array(data_6)
data_7 = np.array(data_7)

# Create a scatter plot
plt.figure(figsize=v)

plt.scatter(dates_log, data_1, marker=',', s=1, color='r', alpha=0.7, label='Trigger asserted')
plt.scatter(dates_log, data_2, marker=',', s=1, color='g', alpha=0.7, label='Trigger rising edges')
plt.scatter(dates_log, data_3, marker=',', s=1, color='b', alpha=0.7, label='Trigger accepted')

x_min_temp = min(dates_log)
x_max_temp = max(dates_log)

# Set the x-limits to be tight to your data_temp
plt.xlim(x_min_temp, x_max_temp)
plt.ylim(trb_trigger_min_cts_s, trb_trigger_max_cts_s)

# Customize the plot
plt.xlabel('Date (YYYY-MM-DD)')
plt.ylabel('TRB rate (cts/s)')
plt.title(f'Trigger TRB rate vs. Date from\n{x_min_temp} to {x_max_temp}')
plt.grid(True)

plt.xticks(rotation=45)

plt.legend()

# Rotate x-axis labels for better readability (optional)

# Show the plot
plt.tight_layout()
plt.savefig(f'trb_trigger_rates_from_{x_min_temp}_to_{x_max_temp}.pdf', format="pdf")
if show_plots: plt.show(); plt.close()


# -----------------------------------------------------------------------------
# 6.1. TRB coincidence rate ---------------------------------------------------
# -----------------------------------------------------------------------------

# Create a scatter plot
plt.figure(figsize=v)

plt.scatter(dates_log, data_4, marker=',', s=1, color='c', alpha=0.7, label='Coincidence Module 0')
plt.scatter(dates_log, data_5, marker=',', s=1, color='m', alpha=0.7, label='Coincidence Module 1')
plt.scatter(dates_log, data_6, marker=',', s=1, color='y', alpha=0.7, label='Coincidence Module 2')
plt.scatter(dates_log, data_7, marker=',', s=1, color='k', alpha=0.7, label='Coincidence Module 3')

x_min_temp = min(dates_log)
x_max_temp = max(dates_log)

# Set the x-limits to be tight to your data_temp
plt.xlim(x_min_temp, x_max_temp)
plt.ylim(trb_coincidence_min_cts_s, trb_coincidence_max_cts_s)

# Customize the plot
plt.xlabel('Date (YYYY-MM-DD)')
plt.ylabel('TRB rate (cts/s)')
plt.title(f'Coincidence TRB rate vs. Date from\n{x_min_temp} to {x_max_temp}')
plt.grid(True)

plt.xticks(rotation=45)

plt.legend()

# Rotate x-axis labels for better readability (optional)

# Show the plot
plt.tight_layout()
plt.savefig(f'trb_coincidence_rates_from_{x_min_temp}_to_{x_max_temp}.pdf', format="pdf")
if show_plots: plt.show(); plt.close()


# -----------------------------------------------------------------------------
# 7. Filtered rates ----------------------------------------------------------
# -----------------------------------------------------------------------------

# Create the concatenated rates file joining the filtered rates from different
# result directories

# Define the root directory
file_path_data = '../../DAQ_data_analysis_results'

# Specify the pattern for files you want to concatenate
file_pattern = 'rates'

# Specify the name of the output text file
output_file = 'concatenated_rates.txt'

with open(output_file, 'w', encoding='utf-8') as f:
    print('----------------------------------------------------------------------', file=f)
    print('Rates concatenated', file=f)
    print('----------------------------------------------------------------------', file=f)

# Function to concatenate files
def concatenate_files(directory, output_file):
    with open(output_file, 'a', encoding='utf-8') as outfile:
        for root, _, files in os.walk(directory):
            for file in files:
                if file.startswith(file_pattern):
                    file_path = os.path.join(root, file)
                    with open(file_path, 'r', encoding='utf-8') as infile:
                        outfile.write(infile.read())

print('--------------------------------------------------------------')
# Check if the root directory exists
if os.path.exists(file_path_data):
    # Iterate through subdirectories and concatenate files
    for directory_name in os.listdir(file_path_data):
        directory_path = os.path.join(file_path_data, directory_name)
        if os.path.isdir(directory_path):
            concatenate_files(directory_path, output_file)

    print(f'Concatenated files starting with "{file_pattern}" into {output_file}')
else:
    print(f'The specified root directory "{file_path_data}" does not exist.')
    
    
# Initialize empty lists to store data

dates = []
data_1 = []
data_2 = []
data_3 = []

# File path
file_path = 'concatenated_rates.txt'  # Replace 'your_file.csv' with the actual file path

with open(file_path, 'r') as file:
    next(file)  # Skip the first line
    
    for line in file:
        if "Rates" in line:  # Check if the line contains the word "Rates"
            continue  # Skip this line and move to the next line
            
        columns = line.strip().split()  # Assuming space-separated columns
        
        # Assuming the second and third columns are in the format 'YYYY-MM-DD'
        date_str = columns[1:3]  # Second column
        date_str = " ".join(date_str)
        
        date = import_date(date_str)
        
        if date is None:
            continue
        
        if start_date <= date <= end_date:
            # Seventh column as data (assuming it's numeric)
            try:
                datum_1 = float(columns[6])
                datum_2 = float(columns[7])
                datum_3 = float(columns[8])
            except IndexError:
                continue
        
            dates.append(date)
            data_1.append(datum_1)
            data_2.append(datum_2)
            data_3.append(datum_3)


# Convert lists to numpy arrays for plotting
dates = np.array(dates)

data_1 = np.array(data_1)
data_2 = np.array(data_2)
data_3 = np.array(data_3)

filtered_rate_dates = dates
filtered_rate = data_1
strongly_filtered_rate = data_3

x_min = min(dates)
x_max = max(dates)

# Preparing the rates to plot.

plt.figure(figsize=v)

color='black'
plt.xlabel('Date (YYYY-MM-DD)')
plt.ylabel('Rate (cts/hr)', color=color)
plt.scatter(dates, data_1, marker=',', s=size, color='r', alpha=0.7, label='Filtered rates')
plt.scatter(dates, data_3, marker=',', s=size, color='b', alpha=0.7, label='Strongly filtered rates')
plt.tick_params(axis='y', labelcolor=color)
plt.xlim(x_min, x_max)
plt.ylim(filtered_rate_min_cts_hr, filtered_rate_max_cts_hr)

plt.grid()
plt.xticks(rotation=45)

plt.legend()
plt.title(f'Rate vs. Date from\n{x_min} to {x_max}')
name = 'rate'

plt.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig(f'{name}_from_{x_min}_to_{x_max}.pdf', format="pdf")
if show_plots: plt.show(); plt.close()


plt.figure(figsize=v)

color='black'
plt.xlabel('Date (YYYY-MM-DD)')
plt.ylabel('Rate (cts/hr)', color=color)
#plt.scatter(dates, data_1, marker=',', s=size, color='r', alpha=0.7, label='Filtered rates')
plt.scatter(dates, data_2, marker=',', s=size, color='g', alpha=0.7, label='Raw rates')
#plt.scatter(dates, data_3, marker=',', s=size, color='b', alpha=0.7, label='Strongly filtered rates')
plt.tick_params(axis='y', labelcolor=color)
plt.xlim(x_min, x_max)
# plt.ylim(11000, 16000)
plt.grid()
plt.xticks(rotation=45)

plt.legend()
plt.title(f'Raw rate vs. Date from\n{x_min} to {x_max}')
name = 'rate_raw'

plt.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig(f'{name}_from_{x_min}_to_{x_max}.pdf', format="pdf")
if show_plots: plt.show(); plt.close()



# -----------------------------------------------------------------------------
# 8. Charge stuff -------------------------------------------------------------
# -----------------------------------------------------------------------------

# Create the concatenated rates file joining the filtered rates from different
# result directories

# Define the root directory

# A function to plot the four RPCs: T1, T2, T3 and T4
def plot_charges(RPC):
    file_path_data = '../../DAQ_data_analysis_results'
    
    file_pattern = f'charges_in_{RPC}'
    
    # Specify the name of the output text file
    output_file = f'concatenated_charges_in_{RPC}.txt'
    
    with open(output_file, 'w', encoding='utf-8') as f:
        print('----------------------------------------------------------------------', file=f)
        print('Charges concatenated', file=f)
        print('----------------------------------------------------------------------', file=f)
    
    # Function to concatenate files
    def concatenate_files(directory, output_file):
        with open(output_file, 'a', encoding='utf-8') as outfile:
            for root, _, files in os.walk(directory):
                for file in files:
                    if file.startswith(file_pattern):
                        file_path = os.path.join(root, file)
                        with open(file_path, 'r', encoding='utf-8') as infile:
                            outfile.write(infile.read())
    
    print('--------------------------------------------------------------')
    # Check if the root directory exists
    if os.path.exists(file_path_data):
        # Iterate through subdirectories and concatenate files
        for directory_name in os.listdir(file_path_data):
            directory_path = os.path.join(file_path_data, directory_name)
            if os.path.isdir(directory_path):
                concatenate_files(directory_path, output_file)
    
        print(f'Concatenated files starting with "{file_pattern}" into {output_file}')
    else:
        print(f'The specified root directory "{file_path_data}" does not exist.')
    
    
    # Initialize lists for data storage
    dates = []
    data_1 = [] # Minimum
    data_2 = [] # 0.05 quantile
    data_3 = [] # Median
    data_4 = [] # Mean
    data_5 = [] # 0.95 quantile
    data_6 = [] # Maximum
    data_7 = [] # Standard Deviation
    
    # File path
    file_path = f'concatenated_charges_in_{RPC}.txt'  # Replace 'your_file.csv' with the actual file path
    
    with open(file_path, 'r') as file:
        next(file)  # Skip the first line
        
        for line in file:
            if "Charges" in line:  # Check if the line contains the word "Rates"
                continue  # Skip this line and move to the next line
                
            columns = line.strip().split()  # Assuming space-separated columns
            
            # Assuming the second and third columns are in the format 'YYYY-MM-DD'
            date_str = columns[1:3]  # Second and third columns
            date_str = " ".join(date_str)
            
            date = import_date(date_str)
            if date is None:
                continue
            
            if start_date <= date <= end_date:
                # Columns 6 to 12 as data (assuming they're numeric)
                try:
                    datum_1 = float(columns[6])
                    datum_2 = float(columns[7])
                    datum_3 = float(columns[8])
                    datum_4 = float(columns[9])  # New data vector
                    datum_5 = float(columns[10])  # New data vector
                    datum_6 = float(columns[11])  # New data vector
                    datum_7 = float(columns[12])  # New data vector
                except IndexError:
                    continue
            
                dates.append(date)
                data_1.append(datum_1)
                data_2.append(datum_2)
                data_3.append(datum_3)
                data_4.append(datum_4)  # Append new data to respective vectors
                data_5.append(datum_5)
                data_6.append(datum_6)
                data_7.append(datum_7)
    
    # Convert lists to numpy arrays for plotting
    dates = np.array(dates)
    data_1 = np.array(data_1)
    data_2 = np.array(data_2)
    data_3 = np.array(data_3)
    data_4 = np.array(data_4)  # Convert new data vectors to numpy arrays
    data_5 = np.array(data_5)
    data_6 = np.array(data_6)
    data_7 = np.array(data_7)
    
    x_min = min(dates)
    x_max = max(dates)
    
    # Preparing the rates to plot.
    plt.figure(figsize=v)  # Set the figure size
    
    size_charge = 25
    color = 'black'
    plt.xlabel('Date (YYYY-MM-DD)')
    plt.ylabel('Charge (AU)', color=color)
    plt.scatter(dates, data_5, marker='^', s=size_charge, color='r', alpha=0.9, label='Quantile 95%')
    plt.errorbar(dates, data_4, yerr=data_7, fmt='ro', alpha=0.7, markersize=size_charge/5, capsize=3, ecolor = 'r', label='Mean with standard deviation')
    plt.scatter(dates, data_3, marker='x', s=size_charge, color='r', alpha=1, label='Median')
    plt.scatter(dates, data_2, marker='v', s=size_charge, color='r', alpha=0.9, label='Quantile 5%')
    
    plt.tick_params(axis='y', labelcolor=color)
    plt.xlim(x_min, x_max)
    plt.ylim(charge_min_AU,charge_max_AU)
    plt.grid()
    plt.xticks(rotation=45)
    
    plt.legend()
    plt.title(f'Charge in {RPC} vs. Date from\n{x_min} to {x_max}')
    name = f'charge_in_{RPC}'
    
    plt.tight_layout()
    plt.savefig(f'{name}_from_{x_min}_to_{x_max}.pdf', format="pdf")
    if show_plots: plt.show(); plt.close()

plot_charges('T1')
plot_charges('T2')
plot_charges('T3')
plot_charges('T4')

# -----------------------------------------------------------------------------
# 9. Efficiencies -------------------------------------------------------------
# -----------------------------------------------------------------------------

# Create the concatenated rates file joining the filtered rates from different
# result directories

# Define the root directory
file_path_data = '../../DAQ_data_analysis_results'

# Specify the pattern for files you want to concatenate
file_pattern = 'efficiencies'

# Specify the name of the output text file
output_file = 'concatenated_efficiencies.txt'

with open(output_file, 'w', encoding='utf-8') as f:
    print('----------------------------------------------------------------------', file=f)
    print('Efficiencies concatenated', file=f)
    print('----------------------------------------------------------------------', file=f)

# Function to concatenate files
def concatenate_files(directory, output_file):
    with open(output_file, 'a', encoding='utf-8') as outfile:
        for root, _, files in os.walk(directory):
            for file in files:
                if file.startswith(file_pattern):
                    file_path = os.path.join(root, file)
                    with open(file_path, 'r', encoding='utf-8') as infile:
                        outfile.write(infile.read())

print('--------------------------------------------------------------')
# Check if the root directory exists
if os.path.exists(file_path_data):
    # Iterate through subdirectories and co0ncatenate files
    for directory_name in os.listdir(file_path_data):
        directory_path = os.path.join(file_path_data, directory_name)
        if os.path.isdir(directory_path):
            concatenate_files(directory_path, output_file)

    print(f'Concatenated files starting with "{file_pattern}" into {output_file}')
else:
    print(f'The specified root directory "{file_path_data}" does not exist.')
    
    
# Initialize empty lists to store data
dates = []
data_1 = []
data_2 = []
data_3 = []
data_4 = []

# File path
file_path = 'concatenated_efficiencies.txt'  # Replace 'your_file.csv' with the actual file path

with open(file_path, 'r') as file:
    next(file)  # Skip the first line
    
    for line in file:
        if "Efficiencies" in line:  # Check if the line contains the word "Rates"
            continue  # Skip this line and move to the next line
            
        columns = line.strip().split()  # Assuming space-separated columns
        
        # Assuming the second and third columns are in the format 'YYYY-MM-DD'
        date_str = columns[1:3]  # Second column
        date_str = " ".join(date_str)
        
        date = import_date(date_str)
        if date is None:
            continue
        
        if start_date <= date <= end_date:
            # Seventh column as data (assuming it's numeric)
            try:
                datum_1 = float(columns[6])
                datum_2 = float(columns[7])
                datum_3 = float(columns[8])
                datum_4 = float(columns[9])
            except IndexError:
                continue
        
            dates.append(date)
            data_1.append(datum_1)
            data_2.append(datum_2)
            data_3.append(datum_3)
            data_4.append(datum_4)


# Convert lists to numpy arrays for plotting
dates = np.array(dates)

data_1 = np.array(data_1)
data_2 = np.array(data_2)
data_3 = np.array(data_3)
data_4 = np.array(data_4)

x_min = min(dates)
x_max = max(dates)

# Preparing the rates to plot.

plt.figure(figsize=v)

color='black'
plt.xlabel('Date (YYYY-MM-DD)')
plt.ylabel('Intrinsic efficiency', color=color)
plt.scatter(dates, data_1, marker='o', s=size, color='r', alpha=0.7, label='T1')
plt.scatter(dates, data_2, marker='o', s=size, color='g', alpha=0.7, label='T2')
plt.scatter(dates, data_3, marker='o', s=size, color='b', alpha=0.7, label='T3')
plt.scatter(dates, data_4, marker='o', s=size, color='orange', alpha=0.7, label='T4')
plt.tick_params(axis='y', labelcolor=color)
plt.xlim(x_min, x_max)
plt.ylim(None, 1)
plt.grid()
plt.xticks(rotation=45)

plt.legend()
plt.title(f'Efficiency vs. Date from\n{x_min} to {x_max}')
name = 'efficiency'

plt.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig(f'{name}_from_{x_min}_to_{x_max}.pdf', format="pdf")
if show_plots: plt.show(); plt.close()

# -----------------------------------------------------------------------------
# 10. MEAN MULTIPLICITIES -----------------------------------------------------
# -----------------------------------------------------------------------------

# Create the concatenated rates file joining the filtered rates from different
# result directories

# Define the root directory
file_path_data = '../../DAQ_data_analysis_results'

# Specify the pattern for files you want to concatenate
file_pattern = 'mean_multiplicities'

# Specify the name of the output text file
output_file = 'concatenated_mean_multiplicities.txt'

with open(output_file, 'w', encoding='utf-8') as f:
    print('----------------------------------------------------------------------', file=f)
    print('Mean multiplicities concatenated', file=f)
    print('----------------------------------------------------------------------', file=f)

# Function to concatenate files
def concatenate_files(directory, output_file):
    with open(output_file, 'a', encoding='utf-8') as outfile:
        for root, _, files in os.walk(directory):
            for file in files:
                if file.startswith(file_pattern):
                    file_path = os.path.join(root, file)
                    with open(file_path, 'r', encoding='utf-8') as infile:
                        outfile.write(infile.read())

print('--------------------------------------------------------------')
# Check if the root directory exists
if os.path.exists(file_path_data):
    # Iterate through subdirectories and co0ncatenate files
    for directory_name in os.listdir(file_path_data):
        directory_path = os.path.join(file_path_data, directory_name)
        if os.path.isdir(directory_path):
            concatenate_files(directory_path, output_file)

    print(f'Concatenated files starting with "{file_pattern}" into {output_file}')
else:
    print(f'The specified root directory "{file_path_data}" does not exist.')
    
    
# Initialize empty lists to store data
dates = []
data_1 = []
data_2 = []
data_3 = []
data_4 = []

# File path
file_path = output_file  # Replace 'your_file.csv' with the actual file path

with open(file_path, 'r') as file:
    next(file)  # Skip the first line
    
    for line in file:
        if "Mean" in line:  # Check if the line contains the word "Rates"
            continue  # Skip this line and move to the next line
            
        columns = line.strip().split()  # Assuming space-separated columns
        
        # Assuming the second and third columns are in the format 'YYYY-MM-DD'
        date_str = columns[1:3]  # Second column
        date_str = " ".join(date_str)
        
        date = import_date(date_str)
        if date is None:
            continue
        
        if start_date <= date <= end_date:
            # Seventh column as data (assuming it's numeric)
            try:
                datum_1 = float(columns[12])
                datum_2 = float(columns[13])
                datum_3 = float(columns[14])
                datum_4 = float(columns[15])
            except IndexError:
                continue
        
            dates.append(date)
            data_1.append(datum_1)
            data_2.append(datum_2)
            data_3.append(datum_3)
            data_4.append(datum_4)


# Convert lists to numpy arrays for plotting
dates = np.array(dates)

data_1 = np.array(data_1)
data_2 = np.array(data_2)
data_3 = np.array(data_3)
data_4 = np.array(data_4)

try:
    x_min = min(dates)
except ValueError:
    print(f"No date was found in '{output_file}'.")
else:
    x_max = max(dates)
    
    # Preparing the rates to plot.
    
    plt.figure(figsize=v)
    
    color='black'
    plt.xlabel('Date (YYYY-MM-DD)')
    plt.ylabel('Mean multiplicity', color=color)
    plt.scatter(dates, data_1, marker='o', s=size, color='r', alpha=0.7, label='T1')
    plt.scatter(dates, data_2, marker='o', s=size, color='g', alpha=0.7, label='T2')
    plt.scatter(dates, data_3, marker='o', s=size, color='b', alpha=0.7, label='T3')
    plt.scatter(dates, data_4, marker='o', s=size, color='orange', alpha=0.7, label='T4')
    plt.tick_params(axis='y', labelcolor=color)
    plt.xlim(x_min, x_max)
    plt.ylim(mean_multiplicity_min,mean_multiplicity_max)
    plt.grid()
    plt.xticks(rotation=45)
    
    plt.legend()
    plt.title(f'Mean multiplicity (if n>0) vs. Date from\n{x_min} to {x_max}')
    name = 'mean_multiplicity'
    
    plt.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig(f'{name}_from_{x_min}_to_{x_max}.pdf', format="pdf")
    if show_plots: plt.show(); plt.close()

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Joined plots ----------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# TRB rate and temperature ----------------------------------------------------
# -----------------------------------------------------------------------------
name = 'joined_temp_and_filtered_rate'

fig, ax1 = plt.subplots(figsize=v)

color = 'tab:blue'
ax1.set_xlabel('Date (YYYY-MM-DD)')
ax1.set_ylabel('Rate (cts/hr)', color=color)  # we already handled the x-label with ax1
ax1.scatter(filtered_rate_dates, filtered_rate, marker='.', s=size, color=color, alpha=0.7, label='Filtered rates')
# ax1.scatter(filtered_rate_dates, strongly_filtered_rate, marker='o', s=size, color='b', alpha=0.7, label='Strongly filtered rates')
ax1.tick_params(axis='y', labelcolor=color)
plt.xlim(x_min, x_max)
plt.ylim(filtered_rate_min_cts_hr, filtered_rate_max_cts_hr)
plt.grid()
plt.xticks(rotation=45)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
color = 'tab:red'

ax2.set_ylabel('Temperature (ºC)', color=color)
ax2.scatter(dates_temp, temp_ext, marker='.', s = size, color=color, alpha = 0.7)
ax2.tick_params(axis='y', labelcolor=color)
plt.xlim(x_min, x_max)
plt.ylim(temp_min_C, temp_max_C)
    
plt.suptitle(f'Temperature and Filtered rate vs. Date from\n{x_min} to {x_max}')

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

plt.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig(f'{name}_from_{x_min}_to_{x_max}.pdf', format="pdf")
if show_plots: plt.show();


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Report creation -------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

x = [a for a in os.listdir() if a.endswith(".pdf")]
x = sorted(x)

merger = PdfMerger()

for pdf in x:
    merger.append(open(pdf, 'rb'))

with open(f"report_results_vs_time_from_{start_date}_to_{end_date}.pdf", "wb") as fout:
    merger.write(fout)

print('----------------------------------------------------------------------')
print(f"Results vs. time report stored as 'report_{start_date}_to_{end_date}.pdf'")

print('--------------------------------------------------------------')
print('Results vs time analysis concluded')
print('--------------------------------------------------------------')
