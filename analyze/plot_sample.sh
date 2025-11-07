#!/bin/bash

# --- 配置区 ---
# 你的 Python 脚本路径
PYTHON_SCRIPT='/home/luoew/project/nowcasting/analyze/4-4.plot_sample.py'

# 你的输入文件路径
INPUT_FILE="/home/luoew/model_output/14s/enhanced_s2s/analyze/visualization/combined_predictions.csv"
#"/home/rika/project/2025/wind-nowcasting25/leye16/model_output/enhanced-s2s-batchsize-512-hidden-512-sampling-revised-residual/analyze/visualization/combined_predictions.csv"
#"/home/rika/project/2025/wind-nowcasting25/leye16/model_output/S2S-decom-251011_1104-sampling-revised-residual/analyze/visualization/combined_predictions.csv"

# 总体时间范围
OVERALL_START="2024-05-11 23:30:00"
OVERALL_END="2024-05-14 09:10:00"

# 每张图的时间间隔（分钟）
INTERVAL_MINUTES=10
# --- 配置结束 ---


# 将日期字符串转换为Unix时间戳，便于比较
current_start_ts=$(date -d "$OVERALL_START" +%s)
overall_end_ts=$(date -d "$OVERALL_END" +%s)

echo "脚本开始执行..."
echo "从: $OVERALL_START"
echo "到: $OVERALL_END"
echo "间隔: $INTERVAL_MINUTES 分钟"
echo "-------------------------------------"

# 循环，直到当前开始时间超过了总的结束时间
while [ "$current_start_ts" -lt "$overall_end_ts" ]; do

    # 计算当前时间段的开始和结束时间字符串
    current_start_str=$(date -d "@$current_start_ts" +"%Y-%m-%d %H:%M:%S")
    current_end_ts=$((current_start_ts + INTERVAL_MINUTES * 60))
    current_end_str=$(date -d "@$current_end_ts" +"%Y-%m-%d %H:%M:%S")

    echo ">>> 正在生成图像，时间范围: \"$current_start_str\" 到 \"$current_end_str\""

    # 执行Python脚本，注意参数要用引号括起来
    python "$PYTHON_SCRIPT" \
        --input_file "$INPUT_FILE" \
        --start_time "$current_start_str" \
        --end_time "$current_end_str"

    # 检查Python脚本是否成功执行
    if [ $? -ne 0 ]; then
        echo "!!! Python脚本执行出错，脚本终止。 !!!"
        exit 1
    fi

    echo "--- 图像生成完毕 ---"
    echo ""

    # 更新下一次循环的开始时间戳
    current_start_ts=$current_end_ts
done

echo "-------------------------------------"
echo "所有任务已完成！"