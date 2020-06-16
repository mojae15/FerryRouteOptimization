#!/bin/bash
DEFAULTVALUE=1

DIRECTION=${1:-$DEFAULTVALUE}

if [ $DIRECTION != 1 ] && [ $DIRECTION != 2 ]
then
    echo "Direction must be 1 or 2"
    exit 1
fi

# Build Dataset for the direction
python3 build_dataset.py $DIRECTION

# Run the Route Planning
./rp

# Plot the Path
python3 plot_path.py
