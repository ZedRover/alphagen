#!/bin/bash

# m 迭代次数
# n 并行线程数
m=${1:-20}
n=${2:-7}

# 外层循环控制总共运行的次数
for ((j = 1; j <= m; j++)); do
	echo "Starting run $j of $m"

	# 内层循环启动n个后台进程
	for ((i = 1; i <= n; i++)); do
		echo "  Starting iteration $i of $n in run $j..."
		python train_10d.py -g 3 &
	done

	# 等待所有后台进程完成
	wait

	echo "Run $j completed."
done

echo "All $m runs completed."
