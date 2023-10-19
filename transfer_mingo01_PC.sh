#!/bin/bash

usage() {
  echo "Usage: $0 [-s] [-b] [-h]"
  echo "Options:"
  echo "  No flag: Invalid option."
  echo "  -s: Send the folder to mingo01."
  echo "  -b: Bring the folder from mingo01."
  echo "  -h: Show this help."
  exit 1
}

while getopts "sbh" opt; do
  case $opt in
    s)
      rsync -avz ./files_for_mingo01_PC/ rpcuser@mingo01:/home/rpcuser/caye_software/
      
      ;;
    b)
      rsync -avz rpcuser@mingo01:/home/rpcuser/gate/software/caye_software/ ./files_from_mingo01
      
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
	echo "Invalid option: -$OPTARG"
	usage
fi

