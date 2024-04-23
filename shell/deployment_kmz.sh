#!/bin/bash
# written by L Garzio on Apr 23 2024
# Generate 4 kmz files (deployed, deployed_ts, deployed_uv and deployed_uv_ts)
# for a user-specified glider deployment: glider-YYYYmmddTHHMM

# Source the global bashrc
if [ -f /etc/bashrc ]; then
. /etc/bashrc
fi

# Source the local bashrc
if [ -f ~/.bashrc ]; then
. ~/.bashrc
fi

# to run, command is: ./deployment_kmz.sh glider-YYYYmmddTHHMM

DEPLOYMENT=$1
EXECDIR=/home/glideradm/code/kmz/gliderkmz 
conda activate gliderkmz

for kmltype in deployed deployed_ts deployed_uv deployed_uv_ts
do
    echo "Writing kmz file for: $DEPLOYMENT $kmltype"
    python ${EXECDIR}/gliderkmz.py $DEPLOYMENT -kml_type $kmltype
done
