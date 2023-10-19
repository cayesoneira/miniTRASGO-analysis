#!/bin/bash

if ! pgrep -f "[m]initrasgo_bot.py" > /dev/null; then
    python3 /home/rpcuser/gate/software/caye_software/minitrasgo_bot.py
fi
