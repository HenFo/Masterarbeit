#!/bin/bash

# Loop through all the mp4 files in the directory
for file in $(find . -type f -name '*.mp4'); do
    echo $file
    # Get the base name of the file
    base_name=$(basename "$file" .mp4)
    # Get path of the file
    path=$(dirname "$file")

    # Convert the mp4 file to wav using ffmpeg
    # turn echo off

    ffmpeg -hide_banner -loglevel error -ac 1 -i "$file" "$path/$base_name.wav"
    
    # Delete the mp4 file
    if [ $? -eq 0 ]; then
        rm "$file"
    else
        echo "Failed to convert $file to wav"
    fi
    
done