#!/bin/bash

RESIZE_STRING="$1"
echo "Resizing images to: $RESIZE_STRING"

# Create output directory
mkdir -p resized_images

# Process all image files
for file in *.{jpg,jpeg,png,gif,bmp,tiff,webp}; do
    if [ -f "$file" ]; then
        echo "Processing: $file"
        magick "$file" -resize $RESIZE_STRING -quality 100 "resized_images/$file"
    fi
done
