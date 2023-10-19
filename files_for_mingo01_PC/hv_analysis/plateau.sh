#!/bin/bash

# Modify to perform a personalized study. (init HV, step, end HV [included]), all in kV
# Define the range of HV values
start_hv=5.4
step=0.05
end_hv=5.7

# Create an array of HV values using seq
w=$(seq $start_hv $step $end_hv)
ww=($w)

# A security value to put when the execution ends
safe_hv_value=5.5

# Start of execution
event_count=$(echo "$w" | wc -w)

echo "There will be $event_count hours of plateau analysis:"
echo "${ww[@]} kV"
read -p "Do you want to continue (Y/N)? " answer

# Convert the answer to uppercase for case-insensitive comparison
answer=${answer^^}

if [ "$answer" != "Y" ]; then
    echo "Exiting the script."
    exit 1
fi


# If confirmation is received, the analysis starts:

# Get the process IDs from the output of `pgrep dabc` and store them in an array
pids=($(pgrep dabc))

# Loop through each process ID and kill it using `kill -9`
for pid in "${pids[@]}"; do
	kill -9 "$pid"
done

# Time after stopping the startRun.sh to wait
sleep 30

for v in $w; do
	cd /home/rpcuser/bin/HV/
	./hv -b 0 -I 1 -V $v -on
	echo "V set to $v"

	# Time for the HV to settle
	sleep 300

	cd /home/rpcuser/trbsoft/userscripts/trb/
	./startRun.sh > /dev/null 2>&1 &

	echo 'Run started'
	date

	# Time of measurement at a certain HV
	sleep 1h

	# Get the process IDs from the output of `pgrep dabc` and store them in an array
	pids=($(pgrep dabc))

	# Loop through each process ID and kill it using `kill -9`
	for pid in "${pids[@]}"; do
		kill -9 "$pid"
	done

	echo 'Run stopped'
	date
	sleep 30
done

# We end setting a safe value for the voltage
/home/rpcuser/bin/HV/hv -b 0 -I 1 -V $safe_hv_value -on

# Time for the HV to settle
sleep 300

# And starting the measurement storage to keep going
cd /home/rpcuser/trbsoft/userscripts/trb/
./startRun.sh > /dev/null 2>&1 &

echo 'Plateau measurement ended'
