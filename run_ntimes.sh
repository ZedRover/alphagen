#!/bin/bash
m=${1:-20}
n=${2:-3}

for ((j = 1; j <= m; j++)); do
	echo "Starting run $j of $m"
	for ((i = 1; i <= n; i++)); do
		echo "  Starting iteration $i of $n in run $j..."
		python train_t0.py -g 1 &
	done
	wait

	echo "Run $j completed."
done

echo "All $m runs completed."
