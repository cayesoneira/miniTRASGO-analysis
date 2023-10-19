#!/bin/bash

usage() {
  echo "Usage: $0 [-l] [-h]"
  echo "Options:"
  echo "  No flag: Perform the data bringing and full analysis including measure vs time quantities."
  echo "  -l: Same, but only for the last day fully registered."
  echo "  -h: Show this help message."
  exit 1
}

export QT_WAYLAND_DISABLE_WINDOWDECORATION=1

cd ./analysis_scripts/

while getopts "lh" opt; do
  case $opt in
    l)
	echo "--------------------------------------------------------"
	echo "Performing analysis of the last day of data"
	echo "--------------------------------------------------------"

	# Get the current date in YYMMDD format
	current_date=$(date +'%y%m%d')
	# Calculate the previous days date: start 2 days and 1 day ago
	one_day_before_date=$(date -d "$current_date - 1 day" +'%y%m%d')
	two_days_before_date=$(date -d "$current_date - 2 day" +'%y%m%d')

	echo "**************************************************************"
	echo "Step 1/3. Bringing files and extracting"
	echo "**************************************************************"

	# Brings only the merged file with the date of the previous day and extracts it
	bash bring_and_extract.sh -l

	echo "**************************************************************"
        echo "Step 2/3. Main analysis"
        echo "**************************************************************"

	# Analize only the merged file with the date of the previous day
	python3 main_analysis.py ../Data_and_Results/DAQ_Data/merged-"$one_day_before_date".txt
	#bash compile_analysis.sh ../Data_and_Results/DAQ_Data/merged-$previous_date.txt

	echo "**************************************************************"
        echo "Step 3/3. Plotting measures vs. time from daq data and logs"
        echo "**************************************************************"

	# Get the full execution of the log, rate, eff, etc. vs time script
	python3 measures_vs_time.py $two_days_before_date $one_day_before_date

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
	echo "--------------------------------------------------------"
	echo "Performing analysis of all merged data in mingo computer"
	echo "--------------------------------------------------------"

	echo "**************************************************************"
	echo "Step 1/3. Bringing files and extracting"
	echo "**************************************************************"

	bash bring_and_extract.sh

	echo "**************************************************************"
        echo "Step 2/3. Main analysis"
        echo "**************************************************************"

	# Set the directory path
	data_dir="../Data_and_Results/DAQ_Data/"
	#data_dir="/media/cayesoneira/54DD-D704/Data"

	# Check if the Data directory exists
	if [ -d "$data_dir" ]; then
	  # Loop through all the .txt files in the Data directory
	  for file in "$data_dir"/*.txt; do
	    if [ -f "$file" ]; then
	      echo "Processing $file"
	      # Add your processing commands here

	      python3 main_analysis.py "$file"
	      #bash compile_analysis.sh "$file"

	    fi
	  done
	else
	  echo "Data directory not found."
	fi

	echo "**************************************************************"
        echo "Step 3/3. Plotting measures vs. time from daq data and logs"
        echo "**************************************************************"
	
	# Get the current date in YYMMDD format
	current_date=$(date +'%y%m%d')
	# Calculate the previous days date: start 2 days and 1 day ago
	start_time=$(date -d "$current_date - 100 day" +'%y%m%d')
	
	python3 measures_vs_time.py $start_time $current_date

fi

echo "**************************************************************"
echo "Complete analysis concluded"
echo "**************************************************************"
