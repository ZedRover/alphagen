#!/bin/bash

# 指定运行次数
max_runs=10

# 初始化计数器
count=0

# 循环运行
while [ $count -lt $max_runs ]; do
	# 运行 Python 脚本
	python train_lstm.py -g 1

	# 增加计数器
	count=$((count + 1))

	# 输出当前运行次数（可选）
	echo "Run count: $count"

	# 等待一段时间（可选，单位：秒）
	sleep 1
done
