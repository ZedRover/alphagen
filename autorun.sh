#!/bin/bash

# 定义检查GPU空载的函数
check_gpus_idle() {
    local idle_count=0
    # 获取每个GPU的利用率
    for i in {0..3}; do
        # 如果GPU利用率低于一定阈值，视为空载
        if [[ $(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i $i) -lt 10 ]]; then
            ((idle_count++))
        fi
    done
    # 如果所有GPU都空载，则返回0，否则返回1
    [[ $idle_count -eq 4 ]] && return 0 || return 1
}

# 定义一个函数来检查当前时间是否在晚上6点到早上7点之间
check_time_within_range() {
    local current_hour=$(date +%H)
    # 检查小时是否在18到07之间
    if ((current_hour >= 18 || current_hour < 7)); then
        return 0
    else
        return 1
    fi
}

# 初始化空载时间计数
idle_hours=0

while true; do
    if check_gpus_idle && check_time_within_range; then
        # 如果GPU空载并且当前时间在指定范围内
        ((idle_hours++))
    else
        idle_hours=0 # 重置空载时间计数
    fi

    if [[ $idle_hours -eq 2 ]]; then
        # 如果GPU连续空载两小时，则运行脚本
        /path/to/run_ntimes.sh &
        break # 运行后退出循环
    fi

    sleep $((900)) # 每刻钟检查一次
done
