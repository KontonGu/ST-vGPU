#
# Created/Modified on Sun May 26 2024
#
# Author: KontonGu (Jianfeng Gu)
# Copyright (c) 2024 TUM - CAPS Cloud
# Licensed under the Apache License, Version 2.0 (the "License")
#
### The script creates a FaST scheduler instance for each GPU device

#!/bin/bash

command -v nvidia-smi
if [ $? -ne 0 ]; then
    echo "No GPU available, sleep forever"
    sleep infinity
fi

function trap_ctrlc ()
{
    echo "Ctrl-C caught...performing clean up"
    for pid in $pids; do
        kill $pid
    done
    echo "Doing cleanup"
    exit 0
}

trap "trap_ctrlc" 2

# each physical gpu have a scheduler port
sched_port=52001
window_size=40
# window_size=40

for gpu in $(nvidia-smi --format=csv,noheader --query-gpu=uuid); do
    echo 0 > $1/$gpu
    python3 /scheduler_instance.py /fast_scheduler /gpu_client $gpu $1/$gpu $2 \
    --scheduler_port $sched_port --window_size $window_size --log_dir $3
    pids="$pids $!"
    sched_port=$(($sched_port+1))
done

wait