# Code ↔ Thesis Mapping (Midterm)

| 论文模块 | 代码位置 | 说明 |
|---|---|---|
| 动态属性 `RD_i, DD_i, c_i, w_i` | `env/load_data.py`, `utils/job_dynamic.py`, `env/fjsp_env.py` | 加载/默认生成/写入环境状态 |
| 三元异构图 Job/Operation/Machine | `env/fjsp_env.py`, `model/ternary_hgat_ppo.py` | 环境维护三类节点特征；模型端三类投影 |
| J-O edge | `model/ternary_hgat_ppo.py` (`jo_attn`) | Job 到 Operation 的关系注意力 |
| O-M edge | `env/fjsp_env.py` (`build_legal_actions`), `model/ternary_hgat_ppo.py` (`om_attn`) | O-M候选边由合法动作决定 |
| J-O 指针滑动 | `env/fjsp_env.py` (`ope_step_batch`) | 每个 job 当前工序指针推进 |
| O-M 候选边更新 | `env/fjsp_env.py` (`build_legal_actions`) | 每个 step 重建 legal O-M mask |
| 动作空间与动作掩码 | `PPO_model.py`, `env/fjsp_env.py` | legal O-M flatten 后 softmax，非法动作屏蔽 |
| 奖励函数 | `env/fjsp_env.py` (`_compute_weighted_metrics`, step reward) | `F_real=Σw_i|C_i-DD_i|` + 终止奖励 |
| HGAT-PPO | `model/ternary_hgat_ppo.py`, `train_hgat_ppo.py` | 新模型与最小训练入口 |

## 备注

- baseline 保留：`PPO_model.py` 未删除。
- 新模型命名：`HGATPPO`。
