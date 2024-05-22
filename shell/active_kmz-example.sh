#!/bin/bash
# written by L Garzio on Apr 23 2024
# Generate or update the active_deployments.yml file which specifies
# active glider deployment names and associated track colors.
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
CONFIGDIR=/path/to/config_files
SAVEDIR=/path/to/savedir
conda activate gliderkmz

# create or update the active_deployments.yml file
python ${EXECDIR}/setup_active_deployments_yaml.py $CONFIGDIR
echo "active_deployments.yml file updated in $CONFIGDIR"

for kmltype in deployed deployed_ts deployed_uv deployed_uv_ts
do
    echo "Writing active kmz file for $kmltype"
    python ${EXECDIR}/gliderkmz.py active -kml_type $kmltype -t $TEMPLATES -c $CONFIGDIR -s $SAVEDIR
done
