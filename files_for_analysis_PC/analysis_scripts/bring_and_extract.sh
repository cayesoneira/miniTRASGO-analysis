#!/bin/bash

usage() {
  echo "Usage: $0 [-l] [-h]"
  echo "Options:"
  echo "  No flag: Perform the merged data bringing and extraction from mingo to local storage."
  echo "  -l: Same, but only for the last day fully registered."
  echo "  -h: Show this help message."
  exit 1
}

mingo_direction=minitrasgo.fis.ucm.es

data_dir_log="../Data_and_Results/Log_data"
data_dir_daq_data="../Data_and_Results/DAQ_Data"

while getopts "lh" opt; do
  case $opt in
    l)
    	# Brings the big log files and only the merged daq_data file with the date of the previous day and extracts it

	# Get the current date in YYMMDD format
	current_date=$(date +'%y%m%d')

	# Calculate the previous day's date
	previous_date=$(date -d "$current_date - 1 day" +'%y%m%d')

	password="mingo@1234"
	filedat="merged*"  # Define the variable with double quotes

	# Log data ------------------------------------------------------------
	# The directory where the data will be stored
	data_dir=$data_dir_log

	mkdir -p $data_dir

	echo "Bringing log files -----------------------------"
	sshpass -p "$password" scp rpcuser@$mingo_direction:/home/rpcuser/logs/done/$filedat $data_dir
	echo "Files brought to local storage -----------------"

	# DAQ data ------------------------------------------------------------
	# The directory where the data will be stored
	data_dir=$data_dir_daq_data
	#data_dir="/media/cayesoneira/54DD-D704/DAQ_data"

	filedat="merged-$previous_date*"

	mkdir -p $data_dir

	echo "Bringing DAQ files -----------------------------"
	sshpass -p "$password" scp rpcuser@$mingo_direction:/home/rpcuser/gate/system/devices/TRB3/data/daqData/asci/$filedat $data_dir
	echo "Files brought to local storage -----------------"

	cd $data_dir

	echo "-------------------------------------------"

	echo "$data_dir/$filedat"
	for file in ./merged-$previous_date.txt.tar.gz; do

	  if [ -f "$file" ]; then
	    echo "Extracted ---------------------------------"
	    tar -xvf "$file"
	    rm $file
	    echo "-------------------------------------------"
	  fi
	done

      ;;
    h)
      usage
      ;;
    \?)
      echo "Invalid option: -$OPTARG"
      usage
      ;;
  esac
done

# If no flag is provided, perform the second script's actions by default
if [ $OPTIND -eq 1 ]; then
	password="mingo@1234"
	filedat="merged*"  # Define the variable with double quotes

	# Log data ------------------------------------------------------------
	# The directory where the data will be stored
	data_dir=$data_dir_log

	mkdir -p $data_dir

	echo "Bringing log files -----------------------------"
	sshpass -p "$password" scp rpcuser@$mingo_direction:/home/rpcuser/logs/done/$filedat $data_dir
	echo "Files brought to local storage -----------------"


	# Event data ------------------------------------------------------------
	# The directory where the data will be stored
	data_dir=$data_dir_daq_data
	#data_dir="/media/cayesoneira/54DD-D704/DAQ_data"

	mkdir -p $data_dir

	echo "Bringing DAQ files -----------------------------"
	sshpass -p "$password" scp rpcuser@$mingo_direction:/home/rpcuser/gate/system/devices/TRB3/data/daqData/asci/$filedat $data_dir
	echo "Files brought to local storage -----------------"

	cd $data_dir

	echo "-------------------------------------------"
	for file in ./*.tar.gz; do
	  if [ -f "$file" ]; then
	    echo "Extracted ---------------------------------"
	    tar -xvf "$file"
	    rm $file
	    echo "-------------------------------------------"
	  fi
	done
fi
