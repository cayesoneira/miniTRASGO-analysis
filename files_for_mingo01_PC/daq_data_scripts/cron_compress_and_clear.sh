#!/bin/bash

if [[ "$1" == "-h" ]];then
    echo "Usage:"
    echo "Function to compress every datafile that is not compressed already."
    exit 0
fi

# Specify the directory where your data files are located
data_dir="/media/externalDisk/gate/system/devices/TRB3/data/daqData/asci"

# Check if the directory exists
if [ ! -d "$data_dir" ]; then
  echo "Directory $data_dir does not exist."
  exit 1
fi

# Change to the data directory
cd "$data_dir" || exit 1

# Compress all data files that are not already compressed
echo "**************************"
echo "**************************"
for file in m*; do
	for file in *.dat; do
		if [ -f "$file" ] && [ "${file##*.}" != "gz" ]; then
    			echo "Compressing:"
			tar -czvf "$file.tar.gz" "$file"
    			rm "$file"
			echo "**************************"
  		fi
	done

	for file in *.txt; do
                if [ -f "$file" ] && [ "${file##*.}" != "gz" ]; then
                        echo "Compressing a merged file:"
                        tar -czvf "$file.tar.gz" "$file"
                        rm "$file"
                        echo "**************************"
                fi
        done
done

echo "Compression complete."
echo "**************************"
echo "**************************"
