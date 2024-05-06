#!/bin/bash

# 设置默认值
script_name="train_lstm.py"
gpu_flag=1
max_runs=10
parallel_count=2

# 从命令行参数获取值
while getopts ":s:g:n:p:" opt; do
	case $opt in
	s)
		script_name="$OPTARG"
		;;
	g)
		gpu_flag="$OPTARG"
		;;
	n)
		max_runs="$OPTARG"
		;;
	p)
		parallel_count="$OPTARG"
		;;
	\?)
		echo "Invalid option -$OPTARG" >&2
		;;
	esac
done

# 初始化计数器
count=0

# 循环运行
while [ $count -lt $max_runs ]; do
	# 内部计数器，用于并行运行
	inner_count=0
	while [ $inner_count -lt $parallel_count ]; do
		# 记录开始时间
		start_time=$(date +%s)

		# 在后台运行 Python 脚本
		python $script_name -g $gpu_flag &

		# 记录结束时间并计算运行时间
		end_time=$(date +%s)
		elapsed_time=$((end_time - start_time))

		# 增加内部计数器
		inner_count=$((inner_count + 1))

		# 输出当前内部运行次数和运行时间
		echo "Inner run count: $inner_count"
		echo "Elapsed time: $elapsed_time seconds"

		# 打印分割线
		echo "-------------------------"

		# 等待 5 秒开始下一个并行运行
		sleep 5
	done

	# 增加外部计数器
	count=$((count + 1))

	# 输出当前外部运行次数
	echo "Outer run count: $count"

	# 等待一段时间（可选，单位：秒）
	sleep 1
done
