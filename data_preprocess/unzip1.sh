#!/bin/bash

# --- Configuration ---
# Source directory containing the .zip files
SOURCE_DIR="/home/luoew/stat_data/haomibo/16"
# Parent destination directory where new folders will be created
DEST_DIR="/home/luoew/stat_data/haomibo/16-unzip"

# --- Script Main Body ---

# 1. Check and create the main destination directory if it doesn't exist
echo "Checking parent destination directory: $DEST_DIR"
mkdir -p "$DEST_DIR"
if [ $? -ne 0 ]; then
    echo "ERROR: Could not create directory $DEST_DIR. Please check permissions."
    exit 1
fi

echo "Starting to process .zip files from $SOURCE_DIR..."
echo "=================================================="

# 2. Find all .zip files in the source directory and loop through them
# -maxdepth 1 ensures we only find files in the top-level of SOURCE_DIR
find "$SOURCE_DIR" -maxdepth 1 -type f -name "*.zip" -print0 | while IFS= read -r -d $'\0' zipfile; do
    
    # 3. Get the base filename from the full path (e.g., "DATA-16-20240511-0514.zip")
    filename=$(basename "$zipfile")
    
    # 4. Create the desired folder name by removing the ".zip" extension
    #    This uses shell parameter expansion, which is very efficient.
    foldername="${filename%.zip}"
    
    # 5. Construct the full path for the new target folder
    target_folder_path="$DEST_DIR/$foldername"
    
    # 6. Create the new directory for this specific zip file
    echo "--> Processing: $filename"
    echo "    Creating target folder: $target_folder_path"
    mkdir -p "$target_folder_path"
    
    # 7. Unzip the file directly into the newly created folder
    #    -o: Overwrite existing files without prompting
    #    -d: Specify the destination directory
    unzip -o "$zipfile" -d "$target_folder_path"
    
    # Check if unzip was successful
    if [ $? -eq 0 ]; then
        echo "    Successfully unzipped to $target_folder_path"
    else
        echo "    WARNING: An error occurred while unzipping $filename."
    fi
    echo "--------------------------------------------------"
done

echo "=================================================="
echo "All .zip files have been processed!"