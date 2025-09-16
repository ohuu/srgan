#!/bin/bash

# Process all image files
for file in *.{jpg,jpeg,png,gif,bmp,tiff,webp}; do
    if [ -f "$file" ]; then
        # ext="${file##*.}"
        # filename=$(basename -- "$file" ".$ext")
        # filename_number=$((10#$filename))
        # filename_new_number=$((filename_number + 2650))
        # new_filename=$((${filename##+0} + 2650))
        mv "$file" "00$file"
        # new_file=$(printf "%04d.%s" $inc $ext)
        # echo "Renaming: $file to $new_file"
        # mv "$file" "./$new_file";
    fi
done

