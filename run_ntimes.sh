#!/bin/bash

# 设置默认次数为10

g=${1:-1}
n=${2:-10}
# 循环运行命令
for ((i = 1; i <= n; i++)); do
	echo "Running iteration $i..."
	python train_1d.py -g $g
done
