help_message="Usage: $0 <origin_data> <feat.scp> <dest_data(scratch)>
Train language models for Switchboard-1, and optionally for Fisher and \n
web-data from University of Washington.\n
options:
  --help          # print this message and exit
  --weblm DIR     # directory for web-data from University of Washington
";

. utils/parse_options.sh

if [ $# -ne 3 ]; then
  printf "$help_message\n";
  exit 1;
fi

if [ -f "$3" ]; then
    echo removing $3 ...
    rm $3;
fi

#!/bin/bash
while IFS='' read -r line || [[ -n "$line" ]]; do

    original_path=$(echo $line | awk '{ print $2 }')
    cleaned_path=$(echo $line | awk '{ print $2 }' | sed -e "s/^.//g")
    new_path=$2/$cleaned_path

    col_1=$(echo $line | awk '{ print $1 }')
    col_3=$(echo $line | awk '{ print $3 }')
    col_4=$(echo $line | awk '{ print $4 }')
    col_5=$(echo $line | awk '{ print $5 }')

    new_line="$col_1 $new_path $col_3 $col_4 $col_5"

    if [ ! -d "$(dirname $new_path)" ]; then
	mkdir -p "$(dirname $new_path)"
    fi

    cp $original_path $new_path

    echo $new_line >> $3

done < "$1"
