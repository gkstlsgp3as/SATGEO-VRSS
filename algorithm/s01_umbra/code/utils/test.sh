#!/bin/bash
output_file="/mnt/d/ournaslist.txt"
directory_path="/mnt/z/umbra/sar-data/tasks/ad hoc/" #David Fuentes Airbase, Chile
aws s3 ls --no-sign-request "s3://umbra-open-data-catalog/sar-data/tasks/ad hoc/" > /mnt/d/test_shell.txt 
sed -i -e 's/                           PRE //g' -e 's/\///g' /mnt/d/test_shell.txt
cd "$directory_path" && ls > "$output_file"
echo "Directory contents listed in $output_file"