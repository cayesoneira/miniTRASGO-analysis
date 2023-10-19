#!/bin/bash

if [[ "$1" == '-h' ]]; then
  echo "Usage:"
  echo "A function that asks for a date range and joins and compresses all data inside that range."
  exit 0
fi

bash compress_and_clear.sh
cd /media/externalDisk/gate/system/devices/TRB3/data/daqData/asci

pwd

# Function to convert YYMMDD to YYDDD
date_to_yyddd() {
    input_date="$1"
    year="${input_date:0:2}"
    month="${input_date:2:2}"
    day="${input_date:4:2}"
    
    # Use 'date' command to calculate day of the year
    day_of_year=$(date -d "$year-$month-$day" +%j)
    
    # Combine year and day of the year
    yyddd="${year}${day_of_year}"
    
    echo "$yyddd"
}

# Prompt the user for date range in YYDDD format
read -p "Enter start date (YYMMDD): " start_date
read -p "Enter end date (YYMMDD): " end_date

input_date_start="$start_date"  # YYMMDD format
start_date=$(date_to_yyddd "$input_date_start")

input_date_end="$end_date"  # YYMMDD format
end_date=$(date_to_yyddd "$input_date_end")

# Prompt the user for the Trigger type (0, 1, or 2)
echo "Enter Trigger type (0, 1, or 2):"
echo -e "\t - 0: all events"
echo -e "\t - 1: only coincidence events"
echo -e "\t - 2: only self-trigger events"
read trigger_type

# Create an array to store matching files
matching_files=()

# Define the output file
output_file="merged-from-$input_date_start-to-$input_date_end.txt"

if [[ $input_date_start == $input_date_end ]]; then
	output_file="merged-$input_date_start.txt"
fi

# Iterate through files in the directory
for file in *.dat.tar.gz; do
  # Extract the date from the filename using regular expressions
  if [[ "$file" =~ ([0-9]{2}[0-9]{3})[0-9]{6}\.dat ]]; then
    file_date="${BASH_REMATCH[1]}"
    
    # Check if the file date is within the specified range
    if [[ "$file_date" -ge "$start_date" && "$file_date" -le "$end_date" ]]; then
      matching_files+=("$file")
    fi
  fi
done

# Check if any matching files were found
if [ ${#matching_files[@]} -eq 0 ]; then
  echo "No matching files found in the specified date range."
  exit 1
fi

# Create an empty output file
> "$output_file"

echo "***********************************"
# Iterate through matching files and append data to the output file
for file in "${matching_files[@]}"; do
  echo "Found and extracted:"

  if [[ $file == *.tar.gz ]]; then
    # Extract the files from the tar.gz archive
    tar -xzf "$file"

    # Loop through extracted files with .dat extension
    for dat_file in *.dat; do
      # Extract the 7th column and check if it matches the trigger type
      du -h $dat_file
      # Now we replace 0000.0000 by 0
      sed -i 's/0000\.0000/0/g' $dat_file
      echo "Changing the 0000.0000 by 0"
      du -h $dat_file

      awk -v trigger="$trigger_type" '$7 == trigger || trigger == 0' "$dat_file" >> "$output_file"
      rm $dat_file
    done
  elif [[ $file == *.dat ]]; then
      # If the file is already a .dat file, just process it directly
      awk -v trigger="$trigger_type" '$7 == trigger || trigger == 0' "$file" >> "$output_file"
      # Now we replace 0000.0000 by 0
      sed -i 's/0000\.0000/0/g' $dat_file
      echo "Changing the 0000.0000 by 0"
      du -h $dat_file

  fi

  echo "***********************************"
done

# Now we replace 0000.0000 by 0
# sed -i 's/0000\.0000/0/g' $output_file

# Compress the output file and remove it
echo "***********************************"
echo "Merged and compressed file:"
du -h $output_file
tar -czf $output_file.tar.gz $output_file
du -h $output_file.tar.gz
echo "***********************************"
echo "***********************************"
rm $output_file
