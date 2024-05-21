#!/bin/bash
# written by L Garzio on Apr 23 2024
# Generate 4 kmz files (deployed, deployed_ts, deployed_uv and deployed_uv_ts)
# for active glider deployments

# Source the global bashrc
if [ -f /etc/bashrc ]; then
. /etc/bashrc
fi

# Source the local bashrc
if [ -f ~/.bashrc ]; then
. ~/.bashrc
fi

EXECDIR=/path/to/repo
TEMPLATES=/path/to/kml/templates
STHRESHOLDS=/path/to/sensor_thresholds.yml
SAVEDIR=/path/to/savedir
conda activate gliderkmz

for kmltype in deployed deployed_ts deployed_uv deployed_uv_ts
do
    echo "Writing active kmz file for $kmltype"
    python ${EXECDIR}/gliderkmz.py active -kml_type $kmltype -t $TEMPLATES -st $STHRESHOLDS -s $SAVEDIR
done
