#!/bin/bash

# 源目录和目标目录
SOURCE_DIR="/home/luoew/stat_data/haomibo/16-unzip"

# 递归处理目录中的所有zip文件
process_directory() {
    local dir="$1"
    
    # 首先处理当前目录中的所有zip文件
    for zipfile in "$dir"/*.zip; do
        if [ -f "$zipfile" ]; then
            # 获取zip文件所在目录
            zip_dir=$(dirname "$zipfile")
            echo "解压 $zipfile 到 $zip_dir"
            unzip -o "$zipfile" -d "$zip_dir"
            # 删除原zip文件
            rm "$zipfile"
        fi
    done
    
    # 递归处理所有子目录
    for item in "$dir"/*; do
        if [ -d "$item" ]; then
            process_directory "$item"
        fi
    done
}

# 开始处理
echo "开始处理所有zip文件..."
process_directory "$SOURCE_DIR"
echo "所有文件处理完成！" 