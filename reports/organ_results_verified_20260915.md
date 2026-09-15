# Organ-CL 双服务器结果核查（每方法一条）

核查日期：2026-09-15。主要补评产物快照：2026-09-15T12:12:43.922099+00:00。

所有主表数值为包含背景的最终四任务平均 Dice（A-Dice），病例与类别均采用宏平均。每种方法只选一条：五个 baseline 与 DER++ 采用同一 60/60/10/10 预算批次；ER 使用现有 40/40/40/40 正式实验。此选择预先按批次与预算确定，不按测试分数选优。调参后 40 epochs 的基线另保留在完整汇总中。

此前表格沿用了较早补评快照，并非五个基线没有结果。现已重新检索原训练汇总、阶段 checkpoint 与含背景补评 JSON。原首个补评任务已结束，后续任务持续产出；无需重训。

| 方法 | 含背景 A-Dice | 各任务 epochs | 证据 |
| --- | ---: | --- | --- |
| Dense Sequential | 0.598730 | 60/60/10/10 | r003 |
| PCE Sequential | 0.614078 | 60/60/10/10 | r004 |
| ZS Sequential | 0.587118 | 60/60/10/10 | r007 |
| ZS-EWC | 0.603900 | 60/60/10/10 | r005 |
| ZS-GPM | 0.620148 | 60/60/10/10 | r006 |
| ZS-DER++ | 0.848059 | 60/60/10/10 | r001 |
| ZS-ER | 0.784476 | 40/40/40/40 | ER |

**7 种方法已有完整四任务的含背景结果。ZS-DER（不带 ++）的后续正式 run 仍是失败记录，不能用 smoke 检查或 DER++ 代替。**

## 两台服务器的实际检索

| 服务器 | 检索位置（相对对应存储根） | 发现与处理 |
| --- | --- | --- |
| jiangsuiyang | `ScribbleCL/organ_comparisons_20260910/runs` | Dense、PCE、ZS Sequential、EWC、GPM 均有四阶段 summary 和 s04.pt；含背景复评全部通过，写入主表。 |
| jiangsuiyang | `ScribbleCL/organ_baseline_tuning_20260911/runs` | 三种基线各有调参后正式四阶段训练结果；EWC/GPM 含背景复评已完成，Sequential 的补评在本快照中仍运行。 |
| jiangsuiyang | `ScribbleCL/organ_T4_from_T3best_lr006_spatial_pair10_20260910`、`ScribbleCL/replay_comparisons_20260911` | 找到正式 DER++、ER 及其含背景结果；DER++ spatial0 进入主表。 |
| jiangsuiyang | `/home/jiangsuiyang/q1d7f`、`/home/jiangsuiyang/q2m8v`、`/data_nas/jiangsuiyang/runs` | 前两处运行以 Domain/Class 为主；NAS 的 Organ 旧筛选有五个三阶段完成 run，u5k2n 无完整四阶段 summary，不能充当当前完整对照。 |
| my-gpu | `ScribbleCL/organ_runs_20260901/runs` | 找到 8 个旧 run 目录；目前只有 manifest 或 T1 best checkpoint，未见完整四阶段 summary。 |
| my-gpu | `ScribbleCL/scribblecl_domain_organ_20260809/runs/organ` | PCE-FT 有四阶段完成结果，但 manifest 标记 resunet32、diagnostic_resunet_only、include_in_final_tables=false；与当前骨干不同，保留为历史诊断。PCE-EWC 文件仅到阶段 2。 |
| my-gpu | `ScribbleCL/core_runs`、`ScribbleCL/outputs` | 发现 Class 运行与标注产物，没有新增可纳入本表的 Organ 四阶段结果。 |

本次检索覆盖上述已知项目与历史运行目录，并未扫描无关项目或声称全服务器不存在其他文件。历史 ResUNet PCE-FT 的含背景平均为 0.525426，仅作检索证据，不计为当前 PCE 方法的另一条正式结果。

## 主表 checkpoint 与原始指标证据

下列路径相对 jiangsuiyang 的 `/data_nas/jiangsuiyang/ScribbleCL/`。六个复评结果的前景重放最大绝对差均为 0；ER 使用原 summary 中显式保存的含背景阶段矩阵。

| 方法 | checkpoint | 指标文件 |
| --- | --- | --- |
| Dense Sequential | `organ_comparisons_20260910/runs/dense-sequential/s04.pt` | `background_recovery_20260915/results/r003.json` |
| PCE Sequential | `organ_comparisons_20260910/runs/pce-sequential/s04.pt` | `background_recovery_20260915/results/r004.json` |
| ZS Sequential | `organ_comparisons_20260910/runs/zs-sequential/s04.pt` | `background_recovery_20260915/results/r007.json` |
| ZS-EWC | `organ_comparisons_20260910/runs/zs-ewc/s04.pt` | `background_recovery_20260915/results/r005.json` |
| ZS-GPM | `organ_comparisons_20260910/runs/zs-gpm/s04.pt` | `background_recovery_20260915/results/r006.json` |
| ZS-DER++ | `organ_T4_from_T3best_lr006_spatial_pair10_20260910/runs/spatial0/s04.pt` | `background_recovery_20260915/results/r001.json` |
| ZS-ER | `replay_comparisons_20260911/runs/organ_zs-er/s04.pt` | `replay_comparisons_20260911/runs/organ_zs-er/summary.json` |

完整逐任务结果、调参后正式结果、旧批次及待补评条目见 [全部结果汇总](all_results_background_dice_20260915.md)。
