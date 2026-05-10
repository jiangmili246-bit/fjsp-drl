# README_Thesis

当前 demo 用于中期检查，证明动态扰动、三元异构图、J-O/O-M 边更新、合法动作集合、结果表格和可视化流程已经跑通。

运行命令：

python run_small_demo.py

本地运行后会生成：
- results/graph_debug.txt
- results/demo_metrics.csv
- results/demo_metrics.xlsx
- results/training_curve.png
- results/hetero_graph_before.png
- results/hetero_graph_after.png
- results/gantt_demo.png

当前 demo 的简化点：
- 当前使用简化启发式调度逻辑；
- 当前不是完整 HGAT-PPO 训练闭环；
- 当前可视化主要用于中期功能验证；
- 后续将替换为完整 HGAT-PPO 训练、标准算例批量实验、消融实验和企业实例验证。

Excel 说明：当前环境缺少 openpyxl 时将跳过 xlsx 输出。可安装：pip install openpyxl
