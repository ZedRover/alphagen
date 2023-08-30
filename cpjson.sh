#!/bin/bash

# 源目录
src_dir="./checkpoints/"

# 目标目录
dest_dir="/home/public2/share_yw/checkpoints"

# 创建目标目录，如果它不存在
mkdir -p "$dest_dir"

# 使用 rsync 来复制 .json 文件，并保持目录结构
rsync -avm --include='*.json' -f 'hide,! */' "$src_dir" "$dest_dir"

echo "All .json files from $src_dir have been copied to $dest_dir, preserving the directory structure."

