import pandas as pd
import json
import os
import re  # 导入正则表达式
import numpy as np  # 导入Numpy
from datetime import datetime

# --- 配置 ---
# 使用你新的 197 列 CSV 文件
CSV_FILE_PATH = '/home/luoew/model_output/14s/enhanced_s2s/analyze/change-ratio&amplitude.csv'  # <--- 替换为你新 CSV 文件的路径
OUTPUT_HTML_PATH = '/home/luoew/model_output/14s/enhanced_s2s/analyze/interactive_report_v2.html'  # 输出新的 HTML 文件
# -----------

def create_interactive_html_report(csv_path, output_html_path):
    """
    读取CSV数据并生成一个独立的、交互式的HTML报告文件。
    V2: 动态检测所有指标 (风速, 变率, 幅度)
    """

    print(f"开始处理 {csv_path}...")

    # 1. 读取CSV数据
    try:
        data = pd.read_csv(csv_path, index_col=0)
        # 关键修复：将 np.nan 替换为 None，以便 json.dumps 将其序列化为 'null'
        # Chart.js 会将 'null' 视为空白/断点，而 'NaN' 会导致JS解析失败
        data = data.replace({np.nan: None})
    except FileNotFoundError:
        print(f"错误: 找不到文件 {csv_path}")
        return
    except Exception as e:
        print(f"读取CSV时发生错误: {e}")
        return

    # 2. 动态检测指标和间隔
    all_cols = data.columns
    intervals = set()
    # 通过正则表达式查找所有 '_XXs' 结尾的列来确定间隔
    for col in all_cols:
        match = re.search(r'_(\d+)s$', col)
        if match:
            intervals.add(int(match.group(1)))
    
    sorted_intervals = sorted(list(intervals))
    print(f"检测到 {len(sorted_intervals)} 个间隔: {sorted_intervals}")

    # 3. 创建指标描述符 (用于JS下拉框)
    metric_descriptors = [{"id": "wind_speed", "label": "原始风速", "y_label": "风速 (m/s)"}]
    
    for interval in sorted_intervals:
        metric_descriptors.append({
            "id": f"change_ratio_{interval}s",
            "label": f"{interval}s 变率",
            "y_label": f"{interval}s 变率"
        })
        metric_descriptors.append({
            "id": f"change_amplitude_{interval}s",
            "label": f"{interval}s 幅度",
            "y_label": f"{interval}s 幅度"
        })
        
    # 4. 将DataFrame转换为JSON兼容的列表
    parsed_data = []
    
    # 准备 1-14 的步长
    steps = list(range(1, 15)) 
    
    print("正在解析数据行...")
    for i, (index, row) in enumerate(data.iterrows()):
        try:
            ts_str = str(index)
            # (时间解析逻辑保持不变)
            try:
                ts_obj = datetime.strptime(ts_str, '%Y-%m-%d %H:%M:%S')
                day = ts_obj.strftime('%Y-%m-%d')
                hour = ts_obj.strftime('%H')
                minute = ts_obj.strftime('%M')
                second = ts_obj.strftime('%S')
            except ValueError:
                parts = ts_str.split(' ')
                day = parts[0]
                time_parts = parts[1].split(':')
                hour = time_parts[0]
                minute = time_parts[1]
                second = time_parts[2]

            # --- 新的数据结构 ---
            # 我们将所有指标打包到一个 'metrics' 字典中
            metrics_dict = {}

            # 4.1 添加 "原始风速"
            true_cols = [f'true_t{t}' for t in steps]
            pred_cols = [f'pred_t{t}' for t in steps]
            metrics_dict["wind_speed"] = {
                "true": row[true_cols].values.tolist(),
                "pred": row[pred_cols].values.tolist()
            }

            # 4.2 循环添加所有变率和幅度
            for interval in sorted_intervals:
                interval_s = f"{interval}s"
                
                # 变率
                ratio_id = f"change_ratio_{interval_s}"
                ratio_true_cols = [f'change_ratio_true_t{t}_{interval_s}' for t in steps]
                ratio_pred_cols = [f'change_ratio_pred_t{t}_{interval_s}' for t in steps]
                
                # 幅度
                amp_id = f"change_amplitude_{interval_s}"
                amp_true_cols = [f'change_amplitude_true_t{t}_{interval_s}' for t in steps]
                amp_pred_cols = [f'change_amplitude_pred_t{t}_{interval_s}' for t in steps]

                # (我们假设所有列都存在，因为是由上一个脚本生成的)
                metrics_dict[ratio_id] = {
                    "true": row[ratio_true_cols].values.tolist(),
                    "pred": row[ratio_pred_cols].values.tolist()
                }
                metrics_dict[amp_id] = {
                    "true": row[amp_true_cols].values.tolist(),
                    "pred": row[amp_pred_cols].values.tolist()
                }
            # --- 结束 ---

            row_data = {
                "originalIndex": i,
                "timestamp": ts_str,
                "day": day,
                "hour": hour,
                "minute": minute,
                "second": second,
                "metrics": metrics_dict  # 替换旧的 trueValues/predValues
            }
            parsed_data.append(row_data)
        except KeyError as e:
            print(f"处理行 {index} 时出错：缺少列 {e} (已跳过)")
        except Exception as e:
            print(f"处理行 {index} 时出错 (已跳过): {e}")

    # 5. 将Python数据序列化为JSON字符串
    # (注意: ensure_ascii=False 确保中文正确显示)
    json_data = json.dumps(parsed_data, indent=4)
    json_metrics = json.dumps(metric_descriptors, ensure_ascii=False, indent=4)

    # 6. 定义HTML模板
    # (我们将修改 <script> 和 <div id="chart-controls">)
    html_template = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>风速预测对比报告 (V2)</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {{ font-family: 'Inter', sans-serif; }}
        .select-filter {{
            min-width: 80px;
            padding: 0.5rem;
            border-radius: 0.375rem;
            border: 1px solid #D1D5DB; /* gray-300 */
            background-color: white;
            box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
            margin-right: 0.5rem;
            margin-bottom: 0.5rem; /* 移动端换行 */
        }}
        .nav-btn {{
            padding: 0.5rem 1rem;
            border: 1px solid transparent;
            border-radius: 0.375rem;
            font-weight: 500;
            color: white;
            background-color: #3B82F6; /* blue-500 */
            cursor: pointer;
            margin: 0 0.25rem;
            margin-bottom: 0.5rem; /* 移动端换行 */
        }}
        .nav-btn:hover {{ background-color: #2563EB; /* blue-600 */ }}
        .nav-btn:disabled {{
            background-color: #9CA3AF; /* gray-400 */
            cursor: not-allowed;
            opacity: 0.7;
        }}
    </style>
</head>
<body class="bg-gray-100 min-h-screen p-4 sm:p-8">
    <div class="max-w-6xl mx-auto bg-white shadow-lg rounded-lg p-6">
        <h1 class="text-2xl sm:text-3xl font-bold text-gray-800 mb-6 text-center">风速模型预测与观测对比</h1>
        
        <div id="chart-controls" class="mb-6 flex flex-wrap items-center justify-center">
            
            <label class="block font-medium text-gray-700 mr-2 mb-2">选择指标:</label>
            <select id="metric-select" class="select-filter"></select>

            <label class="block font-medium text-gray-700 mr-2 mb-2 ml-0 sm:ml-4">选择时间:</label>
            <select id="day-select" class="select-filter"></select>
            <select id="hour-select" class="select-filter"></select>
            <select id="minute-select" class="select-filter"></select>
            <select id="timestamp-select" class="select-filter"></select>
            
            <button id="prev-btn" class="nav-btn"> &lt; 上一时刻</button>
            <button id="next-btn" class="nav-btn">下一时刻 &gt; </button>
        </div>
        
        <div class="w-full h-96 sm:h-[500px]">
            <canvas id="windSpeedChart"></canvas>
        </div>
    </div>

    <script>
        // *** 数据已由Python注入 ***
        const parsedData = {json_data};
        const allMetrics = {json_metrics}; // 新增：指标列表
        // **************************

        let chart; // 用于存储Chart.js实例
        const labels = Array.from({{ length: 14 }}, (_, i) => `+${{i + 1}}s`);
        
        // --- DOM 元素 ---
        const metricSelect = document.getElementById('metric-select'); // 新增
        const daySelect = document.getElementById('day-select');
        const hourSelect = document.getElementById('hour-select');
        const minuteSelect = document.getElementById('minute-select');
        const timestampSelect = document.getElementById('timestamp-select');
        const prevBtn = document.getElementById('prev-btn');
        const nextBtn = document.getElementById('next-btn');

        // (辅助函数 getUniqueValues 和 populateSelect 保持不变)
        function getUniqueValues(arr, key) {{
            const values = arr.map(item => item[key]);
            return [...new Set(values)].sort();
        }}
        function populateSelect(selectEl, options, suffix = '') {{
            const currentVal = selectEl.value;
            selectEl.innerHTML = ''; 
            options.forEach(option => {{
                const optionEl = document.createElement('option');
                optionEl.value = option;
                optionEl.textContent = `${{option}}${{suffix}}`;
                selectEl.appendChild(optionEl);
            }});
            if (options.includes(currentVal)) {{
                selectEl.value = currentVal;
            }}
        }}

        // (辅助函数 updateButtonState 保持不变)
        function updateButtonState(currentIndex) {{
            prevBtn.disabled = (currentIndex <= 0);
            nextBtn.disabled = (currentIndex >= parsedData.length - 1);
        }}

        /**
         * 绘制或更新图表 (已修改)
         * @param {{number}} dataIndex - parsedData中的索引
         */
        function drawChart(dataIndex) {{
            // 检查索引是否有效
            if (dataIndex === undefined || dataIndex === null || !parsedData[dataIndex]) {{
                console.warn("无效的数据索引:", dataIndex);
                if(chart) chart.destroy();
                return;
            }}
            
            // --- 关键改动 ---
            const data = parsedData[dataIndex];
            const selectedMetricId = metricSelect.value; // 1. 获取当前选中的指标ID
            const metricData = data.metrics[selectedMetricId]; // 2. 从 metrics 字典中获取数据
            
            // 3. 查找指标的 Y 轴标签
            const selectedMetric = allMetrics.find(m => m.id === selectedMetricId);
            const yAxisLabel = selectedMetric ? selectedMetric.y_label : '值';
            // --- 结束改动 ---

            if (!metricData) {{
                console.error(`指标 '${{selectedMetricId}}' 在索引 ${{dataIndex}} 的数据中未找到。`);
                if(chart) chart.destroy();
                return;
            }}
            
            const ctx = document.getElementById('windSpeedChart').getContext('2d');

            if (chart) {{
                chart.destroy(); // 如果图表已存在，先销毁
            }}

            chart = new Chart(ctx, {{
                type: 'line',
                data: {{
                    labels: labels,
                    datasets: [
                        {{
                            label: '观测值 (True)',
                            data: metricData.true, // 3. 使用 metricData.true
                            borderColor: 'rgb(59, 130, 246)', 
                            backgroundColor: 'rgba(59, 130, 246, 0.1)',
                            fill: false,
                            tension: 0.1,
                            spanGaps: true // 关键：连接 null 值（NaN）
                        }},
                        {{
                            label: '预测值 (Pred)',
                            data: metricData.pred, // 4. 使用 metricData.pred
                            borderColor: 'rgb(239, 68, 68)', 
                            backgroundColor: 'rgba(239, 68, 68, 0.1)',
                            fill: false,
                            tension: 0.1,
                            spanGaps: true // 关键：连接 null 值（NaN）
                        }}
                    ]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {{
                        legend: {{ position: 'top' }},
                        title: {{
                            display: true,
                            text: `预测起始时间: ${{data.timestamp}}`,
                            font: {{ size: 16 }}
                        }}
                    }},
                    scales: {{
                        x: {{ title: {{ display: true, text: '预测未来时间 (s)' }} }},
                        y: {{ 
                            title: {{ display: true, text: yAxisLabel }}, // 5. 使用动态Y轴标签
                            beginAtZero: true // 风速和幅度通常从0开始
                        }}
                    }}
                }}
            }});
            
            updateButtonState(dataIndex);
        }}

        // (辅助函数 updateSelectorsAndChart 保持不变)
        function updateSelectorsAndChart(newIndex) {{
            if (newIndex < 0 || newIndex >= parsedData.length) return; 
            
            const data = parsedData[newIndex];
            
            daySelect.value = data.day;
            
            updateHourOptions(); 
            hourSelect.value = data.hour;
            
            updateMinuteOptions(); 
            minuteSelect.value = data.minute;
            
            updateTimestampOptions(false); // 重新填充秒，但不触发绘图
            timestampSelect.value = newIndex; 
            
            drawChart(newIndex);
        }}
        
        // --- 级联更新函数 (保持不变) ---
        function updateHourOptions() {{
            const selectedDay = daySelect.value;
            const filteredData = parsedData.filter(d => d.day === selectedDay);
            const hours = getUniqueValues(filteredData, 'hour');
            populateSelect(hourSelect, hours, '时');
            updateMinuteOptions(); 
        }}

        function updateMinuteOptions() {{
            const selectedDay = daySelect.value;
            const selectedHour = hourSelect.value;
            const filteredData = parsedData.filter(d => d.day === selectedDay && d.hour === selectedHour);
            const minutes = getUniqueValues(filteredData, 'minute');
            populateSelect(minuteSelect, minutes, '分');
            updateTimestampOptions(true); 
        }}

        function updateTimestampOptions(drawOnComplete = true) {{
            const selectedDay = daySelect.value;
            const selectedHour = hourSelect.value;
            const selectedMinute = minuteSelect.value;
            const filteredData = parsedData.filter(d => 
                d.day === selectedDay && 
                d.hour === selectedHour && 
                d.minute === selectedMinute
            );
            
            const currentVal = timestampSelect.value;
            timestampSelect.innerHTML = '';
            filteredData.forEach(data => {{
                const optionEl = document.createElement('option');
                optionEl.value = data.originalIndex;
                optionEl.textContent = `${{data.second}}秒`;
                timestampSelect.appendChild(optionEl);
            }});

            if (filteredData.some(d => d.originalIndex == currentVal)) {{
                timestampSelect.value = currentVal;
            }}
            
            if (timestampSelect.options.length > 0) {{
                if (!timestampSelect.value) {{
                    timestampSelect.selectedIndex = 0;
                }}
                
                if (drawOnComplete) {{
                    drawChart(parseInt(timestampSelect.value));
                }}
            }} else {{
                if(chart) chart.destroy();
                updateButtonState(-1);
            }}
        }}


        // 页面加载时执行 (已修改)
        window.onload = function() {{
            // 检查两个数据源
            if (parsedData && parsedData.length > 0 && allMetrics && allMetrics.length > 0) {{
                
                // 1. (新增) 填充指标下拉框
                allMetrics.forEach(metric => {{
                    const optionEl = document.createElement('option');
                    optionEl.value = metric.id;
                    optionEl.textContent = metric.label;
                    metricSelect.appendChild(optionEl);
                }});

                // 2. 绑定事件监听
                // (新增) 指标下拉框改变时，重绘当前图表
                metricSelect.addEventListener('change', () => {{
                    const currentIndex = parseInt(timestampSelect.value);
                    if (!isNaN(currentIndex)) {{
                        drawChart(currentIndex);
                    }}
                }});
                
                // (原有) 时间下拉框
                daySelect.addEventListener('change', updateHourOptions);
                hourSelect.addEventListener('change', updateMinuteOptions);
                minuteSelect.addEventListener('change', () => updateTimestampOptions(true));
                timestampSelect.addEventListener('change', (e) => drawChart(parseInt(e.target.value)));

                // (原有) 按钮
                prevBtn.addEventListener('click', () => {{
                    const currentIndex = parseInt(timestampSelect.value);
                    if (currentIndex > 0) {{
                        updateSelectorsAndChart(currentIndex - 1);
                    }}
                }});
                
                nextBtn.addEventListener('click', () => {{
                    const currentIndex = parseInt(timestampSelect.value);
                    if (currentIndex < parsedData.length - 1) {{
                        updateSelectorsAndChart(currentIndex + 1);
                    }}
                }});

                // 3. 初始化：填充"日"下拉框
                const days = getUniqueValues(parsedData, 'day');
                populateSelect(daySelect, days, '');
                
                // 4. 触发第一次级联更新 (并自动绘制第一个图表)
                updateHourOptions();

            }} else {{
                console.error("未能加载数据或指标定义。");
                document.getElementById('chart-controls').innerHTML = "<p class='text-red-500 text-center'>加载数据或指标失败。</p>";
            }}
        }};
    </script>
</body>
</html>
"""
    
    # 7. 将HTML内容写入文件
    try:
        with open(output_html_path, 'w', encoding='utf-8') as f:
            f.write(html_template)
        print(f"\n--- 成功! ---")
        print(f"已生成交互式报告: {os.path.abspath(output_html_path)}")
        print("您现在可以直接在浏览器中打开该HTML文件。")
        
    except Exception as e:
        print(f"写入HTML文件时发生错误: {e}")


if __name__ == "__main__":
    create_interactive_html_report(CSV_FILE_PATH, OUTPUT_HTML_PATH)