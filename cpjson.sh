#!/bin/bash

# 源目录，包含你要复制的文件和文件夹
SOURCE_DIR="./checkpoints"

# 目标目录，你要复制到哪里
DEST_DIR="/home/public2/share_yw/mycheckpoints"

# 检查源目录是否存在
if [ ! -d "$SOURCE_DIR" ]; then
	echo "源目录 $SOURCE_DIR 不存在."
	exit 1
fi

# 创建目标目录如果它不存在
mkdir -p "$DEST_DIR"

# 开始复制操作
find "$SOURCE_DIR" -type d -o -name "*.json" | while read -r src_file; do
	# 创建目标文件/目录的路径
	dest_file="${src_file/#$SOURCE_DIR/$DEST_DIR}"

	# 如果是目录，则创建目录
	if [ -d "$src_file" ]; then
		mkdir -p "$dest_file"
	else
		# 如果是文件，则复制文件
		cp "$src_file" "$dest_file"
	fi
done

echo "复制完成."
