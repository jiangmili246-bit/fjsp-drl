# README_Thesis

本 demo 用于中期功能验证，采用简化启发式调度，但已展示三元异构图中 J-O/O-M 边和合法动作集合在扰动后非空。

## 运行命令

```bash
python run_small_demo.py
```

## 运行产物说明

本仓库不提交 `results/` 下的运行产物。
请在本地运行 `python run_small_demo.py` 后自动生成：

- `results/graph_debug.txt`
- `results/demo_metrics.csv`
- `results/training_curve.png`

## 当前阶段说明

- 原始 `PPO_model.py` 保持不变；
- 当前 demo 为小规模功能验证，不是最终论文正式实验；
- 如果 HGAT-PPO 尚未完整实现，本 demo 使用简化调度逻辑保证流程闭环；
- 后续将继续扩展标准算例、消融实验和企业动态实例验证。
