# 现有实验结果汇总（Dice 包含背景）

汇总日期：2026-09-15（北京时间）。来源包括本地已归档结果、`my-gpu` 与 `jiangsuiyang` 当前可访问实验目录，以及现有公开仓库中的历史含背景复评结果。主表优先采用本次读取的原始汇总产物；后台改进实验状态另见第 5 节。

本文件汇总已有结果，并按后续请求对保留 checkpoint 补评含背景 Dice，不重新训练。已通过前景重放一致性检查的值才替换 NA；后台待完成项仍明确保留。正文给出可读主表，附录保留筛选、诊断、缺项及未形成完整结果的记录。复制的参考目录、发布镜像和每个 epoch 的重复曲线不计作新实验。这里的“已有”是本次可访问产物范围，不代表不可访问、已删除或未归档的实验。

## 1. 统一指标口径

**本文件所有数值 Dice 均包含背景，范围为 0–1，保留 6 位小数。NA 表示缺少足够的含背景分项或对应评估未完成，绝不是 0。**

对每个病例、每个类别分别计算 Dice，再对类别（含背景 0）和病例做宏平均。沿用原评估器的平滑项、空类别处理、预测和标签映射，不将全体像素池化后的 micro Dice 混入。

```text
Dice_c = (2 × intersection_c + epsilon) / (prediction_size_c + label_size_c + epsilon)
Dice_including_background = (Dice_background + sum(Dice_foreground_classes)) / (K + 1)
```

- 二分类 Organ/Domain：`(背景 Dice + 前景 Dice) / 2`。
- Class 的 T1/T2/T3 分别有 3/2/2 个前景类，不能统一使用二分类公式。
- A-Dice：最终阶段各任务含背景 Dice 的等权平均；Class 的 WCD 是对全体 7 个前景类加 1 个背景类单独评估，不能用 A-Dice 代替。
- 主表中的 BWTR 用含背景阶段矩阵重算：旧任务最终值相对首次学完值的相对变化，再求均值。Class RMA 用对应任务学完时的含背景 Dice 除以统一独立训练参考值。
- 原日志中的 `final_seen_mean`、`benchmark_mean`、部分 `matrix.csv` 和 `seen_validation` 是前景口径，本文件不直接复制这些数值。只有显式含背景字段或完整含背景分项可用于填数。
- 指标展示改为包含背景，不改变历史选模策略。多数正式实验仍按原有前景验证指标选择 checkpoint；按测试集选模的演示结果单列。

## 2. Class 持续学习

### 2.1 当前正式测试结果

六个同预算实验来自 `class_comparisons_20260910_neutral_restart`，均已完成 3 阶段、每任务 80 epochs。seed=42，batch=4，LR=0.03，每任务训练切片 1500，spatial=0；ZS 系列 global=0.1，PCE/Dense 为 0。Dense 使用全标注，其余为相应稀疏监督方案。ER、guarded DER 是后续续跑实验，预算单独列出，不能当作完全同预算对照。

| 方法 | 各任务 epochs | T1 | T2 | T3 | A-Dice | WCD | 来源 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| PCE Sequential | 80 | 0.234826 | 0.316118 | 0.575512 | 0.375486 | 0.197122 | S001 |
| Dense Sequential | 80 | 0.243205 | 0.326306 | 0.883726 | 0.484412 | 0.330855 | S002 |
| ZS Sequential | 80 | 0.242685 | 0.325745 | 0.814501 | 0.460977 | 0.306319 | S003 |
| ZS-EWC | 80 | 0.242609 | 0.325594 | 0.770019 | 0.446074 | 0.290850 | S004 |
| ZS-GPM | 80 | 0.242684 | 0.325715 | 0.777250 | 0.448550 | 0.266062 | S005 |
| ZS-DER++-MiB | 80 | 0.790823 | 0.801020 | 0.711149 | 0.767664 | 0.706680 | S006 |
| ZS-ER | 80/80/60 | 0.244155 | 0.327490 | 0.513165 | 0.361603 | 0.172613 | S007 |
| ZS-DER（guarded） | 80/60/60 | 0.243430 | 0.326606 | 0.672246 | 0.414094 | 0.233408 | S008 |

当前同预算组中，ZS-DER++-MiB 的 A-Dice 为 **0.767664**。后续 ER 为 **0.361603**，guarded DER 为 **0.414094**，旧任务最终表现仍弱。含背景均值提高并不表示前景遗忘消失；背景项会明显影响低分模型的均值。

| 方法 | BWTR（含背景） | RMA（含背景） | MPE | DRR | 来源 |
| --- | --- | --- | --- | --- | --- |
| PCE Sequential | -0.595698 | 0.821586 | 0.000000 | 0.000000 | S001 |
| Dense Sequential | -0.678513 | 1.139106 | 0.000000 | 0.000000 | S002 |
| ZS Sequential | -0.641994 | 1.043821 | 0.000000 | 0.000000 | S003 |
| ZS-EWC | -0.638961 | 1.007204 | 0.000000 | 0.000000 | S004 |
| ZS-GPM | -0.632749 | 0.996842 | 0.000000 | 0.010667 | S005 |
| ZS-DER++-MiB | 0.003098 | 0.984454 | 0.000000 | 0.031333 | S006 |
| ZS-ER | -0.594816 | 0.752010 | 0.000000 | 0.033333 | S007 |
| ZS-DER（guarded） | -0.587539 | 0.878904 | 0.000000 | 0.031667 | S008 |

独立参考：T2=0.778618，T3=0.782023，均为 80 epochs，来自同一正式补充批次。RMA 不使用 T1。MPE 是相邻阶段参数增长率的均值；DRR 按各旧任务被后续任务使用过的唯一原始训练切片占比求均值。GPM 的 DRR 记录子空间估计样本使用比例，不代表保存原始图像进行 replay。

ER 的实际预算为 80/80/60，guarded DER 为 80/60/60。续跑时优化器按阶段重新建立，原 checkpoint 缺少 RNG 状态，随机数重置为配置 seed。guarded DER 使用数值稳定修复和 grad-clip=5；原始 DER 的非有限值失败不能与该完成记录混为同一次无修改训练。

### 2.2 含背景阶段矩阵

仅列已学习任务；NA 表示该阶段尚未学习，不表示评估失败。

| 方法 | 阶段 | T1 | T2 | T3 |
| --- | --- | --- | --- | --- |
| PCE Sequential | 1 | 0.650316 | NA | NA |
| PCE Sequential | 2 | 0.238385 | 0.706397 | NA |
| PCE Sequential | 3 | 0.234826 | 0.316118 | 0.575512 |
| Dense Sequential | 1 | 0.874931 | NA | NA |
| Dense Sequential | 2 | 0.240875 | 0.893979 | NA |
| Dense Sequential | 3 | 0.243205 | 0.326306 | 0.883726 |
| ZS Sequential | 1 | 0.767771 | NA | NA |
| ZS Sequential | 2 | 0.240504 | 0.814522 | NA |
| ZS Sequential | 3 | 0.242685 | 0.325745 | 0.814501 |
| ZS-EWC | 1 | 0.767771 | NA | NA |
| ZS-EWC | 2 | 0.240549 | 0.801787 | NA |
| ZS-EWC | 3 | 0.242609 | 0.325594 | 0.770019 |
| ZS-GPM | 1 | 0.767771 | NA | NA |
| ZS-GPM | 2 | 0.241098 | 0.778452 | NA |
| ZS-GPM | 3 | 0.242684 | 0.325715 | 0.777250 |
| ZS-DER++-MiB | 1 | 0.763909 | NA | NA |
| ZS-DER++-MiB | 2 | 0.792177 | 0.824974 | NA |
| ZS-DER++-MiB | 3 | 0.790823 | 0.801020 | 0.711149 |
| ZS-ER | 1 | 0.776903 | NA | NA |
| ZS-ER | 2 | 0.243723 | 0.660127 | NA |
| ZS-ER | 3 | 0.244155 | 0.327490 | 0.513165 |
| ZS-DER（guarded） | 1 | 0.680158 | NA | NA |
| ZS-DER（guarded） | 2 | 0.241760 | 0.699341 | NA |
| ZS-DER（guarded） | 3 | 0.243430 | 0.326606 | 0.672246 |

### 2.3 历史 Class 结果（独立批次）

| 方法 / run | 各任务 epochs | T1 | T2 | T3 | A-Dice | WCD | 来源 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ZS-DER++-MiB / v6c43 | 80/80/80 | 0.796365 | 0.784973 | 0.756710 | 0.779349 | 0.729182 | S011 |
| ZS-MiB（历史） | 150/150/150 | 0.822525 | 0.827743 | 0.767073 | 0.805781 | NA | S012 |

v6c43 采用 2026-09-10 同批次 checkpoint 复评值；早期公开归档的 A-Dice=0.779349 为同一 run 的另一份复评记录，存在约百万分位差异，不额外计为一次实验。历史 ZS-MiB 的 global=1、MiB KD weight=10、150 epochs，与当前 80 epochs/global=0.1 组不同，不能据此排列统一方法名次。

## 3. Organ 持续学习

### 3.1 Organ 主方法与对照总览

ER 的 4 阶段已完成，采用 T1/T3/T4 训练集减半、T2 保持完整的协议，40 epochs/任务、buffer=64。历史 u5k2n 只是 T4 epoch 21 时保存的 partial checkpoint，不能视为完成的正式对照。

| 方法 / run | 状态 / 预算 | T1 | T2 | T3 | T4 | A-Dice | 来源 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ZS-ER | 完成；40/40/40/40 | 0.715004 | 0.642424 | 0.879449 | 0.901029 | 0.784476 | S013 |
| ZS-DER++ / u5k2n | 未完成；T4 epoch 21 | 0.763569 | 0.497127 | 0.920583 | 0.891977 | 0.768314 | S014 |
| ZS-DER++ / spatial0 | 训练完成；60/60/10/10；补评已完成 | 0.814544 | 0.784317 | 0.887446 | 0.905929 | 0.848059 | S015 |
| dense-sequential | 训练完成；60/60/10/10；含背景补评待完成 | NA | NA | NA | NA | NA | S016 |
| pce-sequential | 训练完成；60/60/10/10；含背景补评待完成 | NA | NA | NA | NA | NA | S017 |
| zs-sequential | 训练完成；60/60/10/10；含背景补评待完成 | NA | NA | NA | NA | NA | S018 |
| zs-ewc | 训练完成；60/60/10/10；含背景补评待完成 | NA | NA | NA | NA | NA | S019 |
| zs-gpm | 训练完成；60/60/10/10；含背景补评待完成 | NA | NA | NA | NA | NA | S020 |
| zs-sequential（调参后） | 训练完成；40/40/40/40；含背景补评待完成 | NA | NA | NA | NA | NA | S021 |
| zs-ewc（调参后） | 训练完成；40/40/40/40；含背景补评待完成 | NA | NA | NA | NA | NA | S022 |
| zs-gpm（调参后） | 训练完成；40/40/40/40；含背景补评待完成 | NA | NA | NA | NA | NA | S023 |

因此，Organ-CL 并非只有 ZS-ER 有训练结果。上版将已经具备含背景分项的记录单列，容易造成只有 ER 的误解；现在把已完成训练的主方法和对照统一展示。不同预算仍不能直接作为匹配对照。

2026-09-15 18:45 北京时间实查：原 Organ 背景补评队列仍停留在第一项 r001，后面 27 项尚未启动；该首轮 worker 未使用后来启用的 HDF5 内存缓存，出现大量 NAS 逐切片读取（进程累计读取量约 1 TB），是补评延迟，不能解读为训练失败或方法没有结果。缓存检查确认同一 brain.h5 的 1,223 张测试切片可在 12.7 秒载入，之后抽取三张切片只需 0.0084 秒。主方法同一 checkpoint 的单独缓存复评已于 18:50 完成，耗时约 70 秒，含背景 A-Dice=0.848059；四任务前景重放最大差为 0。原进程保留。cached 批次是同一 checkpoint 的补评重试，不是新训练实验。

| ER 阶段 | T1 | T2 | T3 | T4 |
| --- | --- | --- | --- | --- |
| 1 | 0.809017 | NA | NA | NA |
| 2 | 0.754812 | 0.794554 | NA | NA |
| 3 | 0.695438 | 0.702402 | 0.917629 | NA |
| 4 | 0.715004 | 0.642424 | 0.879449 | 0.901029 |

该 ER 结果的含背景 BWTR=-0.116427。历史 partial 与当前 ER 的训练预算、配置和数据子集不同，不构成匹配对照。

### 3.2 Organ 批次与背景补评状态

以下批次原来只保存前景结果。已完成 checkpoint 补评的条目更新为含背景值，其余保持 NA；没有用前景旧值替代背景缺项。数据子集、任务策略、学习率与预算变化均需保留。

| 批次 / run | 用途 | 完成阶段 | 配置预算 | 含背景 A-Dice | 来源 |
| --- | --- | --- | --- | --- | --- |
| organ_T3_lr006_spatial_pair10_20260910/runs/spatial0 | 已保存阶段测试 | 3 | 60/60/10 | NA | S024 |
| organ_T3_lr006_spatial_pair10_20260910/runs/spatial001 | 已保存阶段测试 | 3 | 60/60/10 | NA | S025 |
| organ_T4_from_T3best_lr006_spatial_pair10_20260910/runs/spatial0 | 已保存阶段测试 | 4 | 60/60/10/10 | 0.848059 | S015 |
| organ_T4_from_T3best_lr006_spatial_pair10_20260910/runs/spatial001 | 已保存阶段测试 | 4 | 60/60/10/10 | NA | S026 |
| organ_baseline_tuning_20260911/runs/zs-ewc_c0 | 筛选验证 | 4 | 10 | NA | S027 |
| organ_baseline_tuning_20260911/runs/zs-ewc_c1 | 筛选验证 | 4 | 10 | NA | S028 |
| organ_baseline_tuning_20260911/runs/zs-ewc_c2 | 筛选验证 | 4 | 10 | NA | S029 |
| organ_baseline_tuning_20260911/runs/zs-ewc_c3 | 筛选验证 | 4 | 10 | NA | S030 |
| organ_baseline_tuning_20260911/runs/zs-ewc_formal | 已保存阶段测试 | 4 | 40 | NA | S022 |
| organ_baseline_tuning_20260911/runs/zs-gpm_c0 | 筛选验证 | 4 | 10 | NA | S031 |
| organ_baseline_tuning_20260911/runs/zs-gpm_c1 | 筛选验证 | 4 | 10 | NA | S032 |
| organ_baseline_tuning_20260911/runs/zs-gpm_c2 | 筛选验证 | 4 | 10 | NA | S033 |
| organ_baseline_tuning_20260911/runs/zs-gpm_c3 | 筛选验证 | 4 | 10 | NA | S034 |
| organ_baseline_tuning_20260911/runs/zs-gpm_formal | 已保存阶段测试 | 4 | 40 | NA | S023 |
| organ_baseline_tuning_20260911/runs/zs-sequential_c0 | 筛选验证 | 4 | 10 | NA | S035 |
| organ_baseline_tuning_20260911/runs/zs-sequential_c1 | 筛选验证 | 4 | 10 | NA | S036 |
| organ_baseline_tuning_20260911/runs/zs-sequential_c2 | 筛选验证 | 4 | 10 | NA | S037 |
| organ_baseline_tuning_20260911/runs/zs-sequential_c3 | 筛选验证 | 4 | 10 | NA | S038 |
| organ_baseline_tuning_20260911/runs/zs-sequential_formal | 已保存阶段测试 | 4 | 40 | NA | S021 |
| organ_comparisons_20260910/runs/dense-sequential | 已保存阶段测试 | 4 | 60/60/10/10 | NA | S016 |
| organ_comparisons_20260910/runs/pce-sequential | 已保存阶段测试 | 4 | 60/60/10/10 | NA | S017 |
| organ_comparisons_20260910/runs/zs-ewc | 已保存阶段测试 | 4 | 60/60/10/10 | NA | S019 |
| organ_comparisons_20260910/runs/zs-gpm | 已保存阶段测试 | 4 | 60/60/10/10 | NA | S020 |
| organ_comparisons_20260910/runs/zs-sequential | 已保存阶段测试 | 4 | 60/60/10/10 | NA | S018 |
| organ_metrics_T134_half_20260909/runs/ind_T4 | 已保存阶段测试 | 1 | 80 | NA | S039 |
| organ_metrics_half_20260909/runs/ind_T3 | 已保存阶段测试 | 1 | 80 | NA | S040 |
| organ_metrics_half_20260909/runs/organ_no_replay | 已保存阶段测试 | 4 | 60 | NA | S041 |
| organ_metrics_half_20260909/runs/organ_retention | 已保存阶段测试 | 4 | 60 | NA | S042 |

预算补充：`organ_comparisons_20260910` 及对应 T4 配对结果为 60/60/10/10；`organ_baseline_tuning_20260911` 为 12 个 10 epochs/任务筛选，加 ZS Sequential/EWC/GPM 三个 40 epochs/任务正式实验（依次选择 c1/c1/c3）。表中“配置预算”来自各自 manifest，继承训练前缀的 run 应按这里列出的实际执行计划理解，不将 manifest 的单一默认 epoch 字段误读为每阶段实际预算。

### 3.3 后续 replay 批次失败状态

| 任务 | 状态 | 记录错误 | 含背景最终 Dice | 来源 |
| --- | --- | --- | --- | --- |
| organ_zs-der | failed | training exited 1 | NA | S043 |
| domain_zs-der | failed | training exited 1 | NA | S043 |
| organ_zs-er | complete | — | 0.784476 | S043 |
| domain_zs-er | failed | training exited 1 | NA | S043 |

失败状态来自该批次最终 progress，更新时间为 2026-09-12 06:56:43 +0800；不是仍在等待的任务。仅有 exit 1 的汇总证据时，不将所有失败统一归因于显存不足。

## 4. Domain 持续学习与 Joint 诊断

A=BIDMC，B=HK，C=ISBI，D=UCL，E=ISBI_1.5，F=I2CVB。以下值来自已归档的含背景 checkpoint 复评文件。DER++、GPM、Joint 的预算不同；短 Joint 只用作诊断，不宣称是已收敛上界。

| 方法 / run | 预算 | A | B | C | D | E | F | 平均 Dice | 来源 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ZS-DER++ / t4m7b | 80 × 6 | 0.827322 | 0.872079 | 0.799374 | 0.615641 | 0.682554 | 0.815428 | 0.768733 | S044 |
| ZS-GPM / m7v2q | 150 × 6 | 0.616881 | 0.668102 | 0.675925 | 0.563415 | 0.621460 | 0.799747 | 0.657588 | S045 |
| ZS-Joint / y9h4m | 5 epochs；短诊断 | 0.803333 | 0.761996 | 0.825177 | 0.794688 | 0.796832 | 0.674033 | 0.776010 | S046 |

| Domain run | 含背景 BWTR | 依据 |
| --- | --- | --- |
| m7v2q | -0.234234 | 最终含背景分项 + s01–s05 对角线补评 |
| t4m7b | -0.087966 | 最终含背景分项 + s01–s05 对角线补评 |

这里可比较保存 checkpoint 的含背景最终分数。已完成对角线补评的模型可重算含背景 BWTR；E-FWT 所需的其余矩阵仍不完整，不复用归档 CSV 中按前景计算的旧值。9 月 11 日新启动的 Domain ER、DER 均失败，见 3.3；不要用它们覆盖这两项历史完成结果。

## 5. GPU 3 改进实验快照

`class_replay_improve_20260915_r2` 是在既有 T1 checkpoint 上进行的短验证筛选，不是新的完整三任务测试结果。筛选门槛沿用原先的前景验证协议；本报告不将门槛分数改名为含背景 Dice。ER 改进包含 partial-background PCE、T1 后冻结 BN 等；DER 另有旧头保护与 feature target 刷新。

本次状态读取时间：**2026-09-15 15:58:33 北京时间**。四个 5-epoch 对照/改进筛选均已完成；ER 通过门槛并已进入 T2 的 60-epoch 续跑，DER 改进未通过原门槛。以下为 progress 记录状态，尚无新的完整三任务结果。

| 作业 | 记录状态 | 来源 |
| --- | --- | --- |
| 总流程 | running | S047 |
| jobs/zs-er-smoke | complete | S047 |
| jobs/zs-der-smoke | complete | S047 |
| jobs/zs-er-control5 | complete | S047 |
| jobs/zs-er-improved5 | complete | S047 |
| jobs/zs-der-control5 | complete | S047 |
| jobs/zs-der-improved5 | complete | S047 |
| jobs/zs-er-T2-60 | running | S047 |

| 筛选 run | 数据用途 | 当前 T2 含背景 Dice | 旧 T1 含背景 Dice | T1/T2 含背景均值 | 来源 |
| --- | --- | --- | --- | --- | --- |
| zs-er-control5 | 验证集 | 0.566031 | 0.243774 | 0.404902 | S048 |
| zs-er-improved5 | 验证集 | 0.677205 | 0.716085 | 0.696645 | S049 |
| zs-der-control5 | 验证集 | 0.514116 | 0.242716 | 0.378416 | S050 |
| zs-der-improved5 | 验证集 | 0.482276 | 0.240082 | 0.361179 | S051 |

原有 T2 值来自 best_validation，旧 T1 只保存前景标量。已完成补评的条目使用同一 s02.pt 在原验证集重新推理，补齐 T1/T2 及均值；未完成的保持 NA。ER 改进筛选通过、DER 改进筛选未通过原门槛，正式提升幅度仍需完成同预算测试后判断。首轮的 int16 gather 索引故障、被修复替代的筛选及 smoke 检查仅列作工程记录。

## 6. 独立训练、参数筛选与演示结果

### 6.1 已归档独立训练测试表

以下 CSV 含有显式 `inclusive_dice`，因此可以保留数值。80 epochs、seed=42；Full/Dense 与 scribble/PCE 的监督强度不同。不同批次的超参数和稀疏标注不保证相同，不能作为统一独立参考随意互换。

**independent_training_80e_seed42_20260906.csv**（S052）

| 场景 | 任务 | 监督 / 方法 | 测试含背景 Dice | 选中 checkpoint 的验证含背景 Dice | 原始 best_epoch（0-based） |
| --- | --- | --- | --- | --- | --- |
| domain | A | full | 0.941271 | 0.935475 | 79 |
| domain | A | scribble | 0.462710 | 0.413963 | 9 |
| domain | B | full | 0.931573 | 0.930966 | 54 |
| domain | B | scribble | 0.578363 | 0.547963 | 54 |
| domain | C | full | 0.949581 | 0.947424 | 54 |
| domain | C | scribble | 0.591954 | 0.695170 | 24 |
| domain | D | full | 0.887690 | 0.873993 | 19 |
| domain | D | scribble | 0.620321 | 0.667773 | 29 |
| domain | E | full | 0.939701 | 0.925546 | 74 |
| domain | E | scribble | 0.600092 | 0.542591 | 4 |
| domain | F | full | 0.929383 | 0.871148 | 29 |
| domain | F | scribble | 0.580432 | 0.619902 | 9 |
| class | T1 | full | 0.891301 | 0.895523 | 64 |
| class | T1 | scribble | 0.670125 | 0.728779 | 34 |
| class | T2 | full | 0.901448 | 0.720966 | 54 |
| class | T2 | scribble | 0.741404 | 0.670359 | 39 |
| class | T3 | full | 0.864248 | 0.863412 | 44 |
| class | T3 | scribble | 0.639552 | 0.627738 | 9 |

**zs_independent_80e_seed42_20260907.csv**（S053）

| 场景 | 任务 | 监督 / 方法 | 测试含背景 Dice | 选中 checkpoint 的验证含背景 Dice | 原始 best_epoch（0-based） |
| --- | --- | --- | --- | --- | --- |
| class | T1 | ZS | 0.757161 | 0.785610 | 64 |
| class | T2 | ZS | 0.775709 | 0.656201 | 19 |
| class | T3 | ZS | 0.805854 | 0.809658 | 74 |
| domain | A | ZS | 0.604066 | 0.557808 | 54 |
| domain | B | ZS | 0.634619 | 0.594458 | 54 |
| domain | C | ZS | 0.715648 | 0.825693 | 74 |
| domain | D | ZS | 0.767361 | 0.808001 | 59 |
| domain | E | ZS | 0.740512 | 0.643054 | 39 |
| domain | F | ZS | 0.640100 | 0.720441 | 24 |

**domain_zs_independent_matched_80e_seed42_20260907.csv**（S054）

| 场景 | 任务 | 监督 / 方法 | 测试含背景 Dice | 选中 checkpoint 的验证含背景 Dice | 原始 best_epoch（0-based） |
| --- | --- | --- | --- | --- | --- |
| domain | A | ZS | 0.419429 | 0.413899 | 59 |
| domain | B | ZS | 0.607988 | 0.577584 | 9 |
| domain | C | ZS | 0.653783 | 0.766919 | 4 |
| domain | D | ZS | 0.513277 | 0.537132 | 4 |
| domain | E | ZS | 0.571057 | 0.513767 | 4 |
| domain | F | ZS | 0.515028 | 0.541856 | 4 |

### 6.2 20-epoch ZS 参数筛选（验证集）

| 场景 | 候选 | PCE weight | global | spatial | 验证含背景 Dice | 原协议选中 | 来源 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| domain | c01 | 1 | 0.1 | 0.01 | 0.687541 | false | S055 |
| domain | c02 | 1 | 0.5 | 0.01 | 0.749696 | false | S055 |
| domain | c03 | 1 | 1 | 0.01 | 0.754226 | false | S055 |
| domain | c04 | 1 | 2 | 0.01 | 0.705839 | false | S055 |
| domain | c05 | 1 | 0.5 | 0.05 | 0.786198 | false | S055 |
| domain | c06 | 1 | 1 | 0.05 | 0.785498 | false | S055 |
| domain | c07 | 0.5 | 1 | 0.05 | 0.799429 | true | S055 |
| domain | c08 | 2 | 1 | 0.05 | 0.755852 | false | S055 |
| class | c01 | 1 | 0.1 | 0.01 | 0.612827 | false | S055 |
| class | c02 | 1 | 0.5 | 0.01 | 0.605503 | false | S055 |
| class | c03 | 1 | 1 | 0.01 | 0.632270 | false | S055 |
| class | c04 | 1 | 2 | 0.01 | 0.584887 | false | S055 |
| class | c05 | 1 | 0.5 | 0.05 | 0.644302 | true | S055 |
| class | c06 | 1 | 1 | 0.05 | 0.611551 | false | S055 |
| class | c07 | 0.5 | 1 | 0.05 | 0.623812 | false | S055 |
| class | c08 | 2 | 1 | 0.05 | 0.597079 | false | S055 |

### 6.3 其他独立训练与筛选（逐 run）

“测试”表示验证选模后已有 test 字段；“验证”仅有 best_validation；“测试选模演示”使用测试数据选择 checkpoint，没有独立 held-out 测试结论。合成与 smoke 记录只检验流程，不用于方法排名。

| 批次 / run | 任务 | 完成 epochs | 用途 | 含背景 Dice | 来源 |
| --- | --- | --- | --- | --- | --- |
| independent_A_spatial_sweep20_formal80_20260907/formal80 | A | 80 | 测试 | 0.876482 | S056 |
| independent_A_spatial_sweep20_formal80_20260907/sweep_s00 | A | 20 | 验证 | 0.752994 | S057 |
| independent_A_spatial_sweep20_formal80_20260907/sweep_s01 | A | 20 | 验证 | 0.801644 | S058 |
| independent_A_spatial_sweep20_formal80_20260907/sweep_s02 | A | 20 | 验证 | 0.733786 | S059 |
| independent_A_spatial_sweep20_formal80_20260907/sweep_s03 | A | 20 | 验证 | 0.744954 | S060 |
| independent_A_spatial_sweep20_formal80_20260907/sweep_s04 | A | 20 | 验证 | 0.755679 | S061 |
| independent_A_spatial_sweep20_formal80_20260907/sweep_s05 | A | 20 | 验证 | 0.767118 | S062 |
| independent_A_warmup_sweep80_20260907/warmup10 | A | 80 | 验证 | 0.835785 | S063 |
| independent_A_warmup_sweep80_20260907/warmup20 | A | 80 | 验证 | 0.795353 | S064 |
| independent_A_warmup_sweep80_20260907/warmup40 | A | 80 | 验证 | 0.824141 | S065 |
| independent_A_warmup_sweep80_20260907/warmup60 | A | 80 | 验证 | 0.833942 | S066 |
| independent_BF_formal80_warmup10_20260907/B | B | 80 | 测试 | 0.812001 | S067 |
| independent_BF_formal80_warmup10_20260907/C | C | 80 | 测试 | 0.798257 | S068 |
| independent_BF_formal80_warmup10_20260907/D | D | 80 | 测试 | 0.740203 | S069 |
| independent_BF_formal80_warmup10_20260907/E | E | 80 | 测试 | 0.835373 | S070 |
| independent_BF_formal80_warmup10_20260907/F | F | 80 | 测试 | 0.783999 | S071 |
| independent_B_shared_smoke_20260907 | B | 6 | 验证 | 0.497092 | S072 |
| independent_demo_smoke_20260907 | A | 2 | 测试选模演示 | 0.304906 | S073 |
| organ_T2_original_scribble_20260908/runs/organ_fg20 | T2 | 80 | 测试 | 0.762081 | S074 |
| organ_T2_reference_recovery_20260908/runs/domain_control | T2 | 80 | 测试 | 0.744635 | S075 |
| organ_T2_reference_recovery_20260908/runs/organ_reference | T2 | 80 | 测试 | 0.732377 | S076 |
| organ_T2_spatial30_deterministic_20260908/runs/domain_control | T2 | 80 | 测试 | 0.742221 | S077 |
| organ_T2_spatial30_deterministic_20260908/runs/organ_reference | T2 | 80 | 测试 | 0.742221 | S078 |
| organ_T2_spatial_start30_20260908/runs/domain_control | T2 | 80 | 测试 | 0.707909 | S079 |
| organ_T2_spatial_start30_20260908/runs/organ_reference | T2 | 80 | 测试 | 0.749531 | S080 |
| spatial_sweep_smoke_20260907 | A | 6 | 验证 | 0.119363 | S081 |
| tune_independent_A_07261_seed42_20260907_1303/independent_pattern_f5_b10 | A | 150 | 测试 | 0.836364 | S082 |
| tune_independent_A_07261_seed42_20260907_1303/smoke | A | 1 | 验证 | 0.105891 | S083 |

### 6.4 测试集选模的演示实验

以下两批的 Dice 都包含背景，但用于演示选模，不是无偏测试成绩。第二批包括 24 个 20-epoch 筛选和 6 个 80-epoch 后续 run；“formal”是原目录阶段名，不能消除测试选模造成的偏差。

| 域 | run | 阶段 | epochs | LR | global | spatial | 演示含背景 Dice | 选中 epoch（1-based） | 来源 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | A | demo | 80 | 见原配置 | 见原配置 | 见原配置 | 0.865714 | 22 | S084 |
| B | B | demo | 80 | 见原配置 | 见原配置 | 见原配置 | 0.764733 | 58 | S084 |
| C | C | demo | 80 | 见原配置 | 见原配置 | 见原配置 | 0.816082 | 44 | S084 |
| D | D | demo | 80 | 见原配置 | 见原配置 | 见原配置 | 0.737451 | 62 | S084 |
| E | E | demo | 80 | 见原配置 | 见原配置 | 见原配置 | 0.840677 | 74 | S084 |
| F | F | demo | 80 | 见原配置 | 见原配置 | 见原配置 | 0.829394 | 64 | S084 |

| 域 | run | 阶段 | epochs | LR | global | spatial | 演示含背景 Dice | 选中 epoch（1-based） | 来源 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | A_s0 | sweep | 20 | 0.03 | 1.0 | 0.01 | 0.871830 | 17 | S085 |
| A | A_s1 | sweep | 20 | 0.01 | 1.0 | 0.01 | 0.847063 | 19 | S085 |
| A | A_s2 | sweep | 20 | 0.03 | 0.1 | 0.01 | 0.877727 | 17 | S085 |
| A | A_s3 | sweep | 20 | 0.01 | 0.1 | 0.01 | 0.805788 | 15 | S085 |
| A | A_formal80 | formal | 80 | 0.03 | 0.1 | 0.01 | 0.787237 | 45 | S085 |
| B | B_s0 | sweep | 20 | 0.03 | 1.0 | 0.01 | 0.770591 | 10 | S085 |
| B | B_s1 | sweep | 20 | 0.01 | 1.0 | 0.01 | 0.740415 | 11 | S085 |
| B | B_s2 | sweep | 20 | 0.03 | 0.1 | 0.01 | 0.748289 | 10 | S085 |
| B | B_s3 | sweep | 20 | 0.01 | 0.1 | 0.01 | 0.729194 | 11 | S085 |
| B | B_formal80 | formal | 80 | 0.03 | 1.0 | 0.01 | 0.770494 | 42 | S085 |
| C | C_s0 | sweep | 20 | 0.03 | 1.0 | 0.01 | 0.753784 | 12 | S085 |
| C | C_s1 | sweep | 20 | 0.01 | 1.0 | 0.01 | 0.794827 | 6 | S085 |
| C | C_s2 | sweep | 20 | 0.03 | 0.1 | 0.01 | 0.784348 | 4 | S085 |
| C | C_s3 | sweep | 20 | 0.01 | 0.1 | 0.01 | 0.735532 | 5 | S085 |
| C | C_formal80 | formal | 80 | 0.01 | 1.0 | 0.01 | 0.818938 | 71 | S085 |
| D | D_s0 | sweep | 20 | 0.03 | 1.0 | 0.01 | 0.726577 | 11 | S085 |
| D | D_s1 | sweep | 20 | 0.01 | 1.0 | 0.01 | 0.697010 | 11 | S085 |
| D | D_s2 | sweep | 20 | 0.03 | 0.1 | 0.01 | 0.691651 | 10 | S085 |
| D | D_s3 | sweep | 20 | 0.01 | 0.1 | 0.01 | 0.642995 | 12 | S085 |
| D | D_formal80 | formal | 80 | 0.03 | 1.0 | 0.01 | 0.765534 | 60 | S085 |
| E | E_s0 | sweep | 20 | 0.03 | 1.0 | 0.01 | 0.809279 | 14 | S085 |
| E | E_s1 | sweep | 20 | 0.01 | 1.0 | 0.01 | 0.810144 | 17 | S085 |
| E | E_s2 | sweep | 20 | 0.03 | 0.1 | 0.01 | 0.805110 | 8 | S085 |
| E | E_s3 | sweep | 20 | 0.01 | 0.1 | 0.01 | 0.728237 | 10 | S085 |
| E | E_formal80 | formal | 80 | 0.01 | 1.0 | 0.01 | 0.848819 | 74 | S085 |
| F | F_s0 | sweep | 20 | 0.03 | 1.0 | 0.01 | 0.773003 | 18 | S085 |
| F | F_s1 | sweep | 20 | 0.01 | 1.0 | 0.01 | 0.820966 | 10 | S085 |
| F | F_s2 | sweep | 20 | 0.03 | 0.1 | 0.01 | 0.789440 | 14 | S085 |
| F | F_s3 | sweep | 20 | 0.01 | 0.1 | 0.01 | 0.800218 | 10 | S085 |
| F | F_formal80 | formal | 80 | 0.01 | 1.0 | 0.01 | 0.837616 | 51 | S085 |

## 7. 其他已有阶段结果与工程诊断

以下为尚未在上述主表逐项列出的原始 summary。数值只能代表保存产物中的评估范围；阶段数少于任务总数、合成数据、短 smoke、验证筛选均不能称为完整正式结果。若最终测试含背景分项缺失，则只显示当前任务 best_validation 的含背景值并明确用途；不会把它当作最终跨任务均值。

| 位置 | 批次 / run | 完成阶段或 epochs | 配置预算 | 评估范围 | 含背景 Dice | 来源 |
| --- | --- | --- | --- | --- | --- | --- |
| my-gpu | class_comparisons_20260910/dense_smoke | 1 | 1 | 工程/短诊断；仅当前任务验证 | 0.244697 | S086 |
| my-gpu | class_comparisons_20260910/gpm_smoke | 2 | 1 | 工程/短诊断；仅当前任务验证 | 0.326683 | S087 |
| my-gpu | class_independent_spatial_sweep_20260908/jobs/cl_spatial_0.01 | 3 | 40 | checkpoint 补评；已见任务验证 | 0.775111 | S088 |
| my-gpu | class_independent_spatial_sweep_20260908/jobs/ind_T1_lr0.01 | 1 | 20 | checkpoint 补评；已见任务验证 | 0.735565 | S089 |
| my-gpu | class_independent_spatial_sweep_20260908/jobs/ind_T1_lr0.03 | 1 | 20 | checkpoint 补评；已见任务验证 | 0.692540 | S090 |
| my-gpu | class_independent_spatial_sweep_20260908/jobs/ind_T2_lr0.01 | 1 | 20 | checkpoint 补评；已见任务验证 | 0.599318 | S091 |
| my-gpu | class_independent_spatial_sweep_20260908/jobs/ind_T2_lr0.03 | 1 | 20 | checkpoint 补评；已见任务验证 | 0.658331 | S092 |
| my-gpu | class_independent_spatial_sweep_20260908/jobs/ind_T3_lr0.01 | 1 | 20 | checkpoint 补评；已见任务验证 | 0.711128 | S093 |
| my-gpu | class_independent_spatial_sweep_20260908/jobs/ind_T3_lr0.03 | 1 | 20 | checkpoint 补评；已见任务验证 | 0.713457 | S094 |
| my-gpu | class_independent_spatial_sweep_20260908/smoke_cl_spatial_original_gco | 3 | 1 | 工程/短诊断；含背景分项缺失 | NA | S095 |
| my-gpu | class_independent_spatial_sweep_20260908/smoke_ind_T3_gco | 1 | 1 | 工程/短诊断；含背景分项缺失 | NA | S096 |
| my-gpu | class_replay_improve_20260915/jobs/zs-der-smoke | 2 | 80/1/60 | 工程/短诊断；仅当前任务验证 | 0.324897 | S097 |
| my-gpu | class_replay_improve_20260915_r2/jobs/zs-der-smoke | 2 | 80/1/60 | 工程/短诊断；仅当前任务验证 | 0.324897 | S098 |
| my-gpu | class_replay_improve_20260915_r2/jobs/zs-er-smoke | 2 | 80/1/60 | 工程/短诊断；仅当前任务验证 | 0.318258 | S099 |
| my-gpu | core_runs/class_q8v2n6_seed42_150e_20260811T1345Z/pce_mib_seed42 | 3 | 150 | checkpoint 补评；测试 | 0.666446 | S100 |
| my-gpu | core_runs/class_q8v2n6_seed42_150e_20260811T1345Z/pce_sequential_seed42 | 3 | 150 | checkpoint 补评；测试 | 0.423118 | S101 |
| my-gpu | core_runs/class_q8v2n6_seed42_150e_20260811T1345Z/zs_mib_global1_origscale_seed42 | 3 | 150 | checkpoint 补评；测试 | 0.805777 | S102 |
| my-gpu | core_runs/class_q8v2n6_seed42_150e_20260811T1345Z/zs_sequential_global1_origscale_seed42 | 3 | 150 | checkpoint 补评；测试 | 0.461417 | S103 |
| my-gpu | domain_joint_runs_20260901/runs/j0b2l15_r1 | — | 150 | 含背景分项缺失 | NA | S104 |
| my-gpu | domain_joint_runs_20260901/runs/j1b4l30_r1 | — | 150 | 含背景分项缺失 | NA | S105 |
| my-gpu | domain_runs_20260901/runs/d0gpm_r1 | 6 | 150 | checkpoint 补评；测试 | 0.469249 | S106 |
| my-gpu | domain_runs_20260901/runs/d3p9n_r1 | 6 | 150 | checkpoint 补评；测试 | 0.451765 | S107 |
| my-gpu | domain_runs_20260901/smoke/joint_smoke_b8 | — | 1 | 工程/短诊断；含背景分项缺失 | NA | S108 |
| my-gpu | scribblecl_domain_organ_20260809/runs/domain/domain_pce_ft_seed42_20260809T082951Z | — | 未记录 | 含背景分项缺失 | NA | S109 |
| my-gpu | scribblecl_domain_organ_20260809/runs/organ/organ_pce_ft_seed42_20260809T083415Z | — | 未记录 | 含背景分项缺失 | NA | S110 |
| jiangsuiyang | class_independent_spatial_sweep_20260908/smoke_cl_spatial_chunked | 3 | 1 | 工程/短诊断；含背景分项缺失 | NA | S111 |
| jiangsuiyang | class_independent_spatial_sweep_20260908/smoke_ind_T3 | 1 | 1 | 工程/短诊断；含背景分项缺失 | NA | S112 |
| jiangsuiyang | organ_CL_throughput_20260908/memory_spatial_off/run | 3 | 1 | 工程/短诊断；含背景分项缺失 | NA | S113 |
| jiangsuiyang | organ_CL_throughput_20260908/memory_spatial_on/run | 2 | 1 | 工程/短诊断；含背景分项缺失 | NA | S114 |
| jiangsuiyang | organ_T13_half_cl_20260908/balance_checks_20260908_v2 | — | 未记录 | 含背景分项缺失 | NA | S115 |
| jiangsuiyang | organ_T13_half_cl_20260908/diagnostic_t2_seed43_v3 | — | 未记录 | 含背景分项缺失 | NA | S116 |
| jiangsuiyang | organ_T13_half_cl_20260908/formal_small_alpha_20260908 | 3 | 60 | 含背景分项缺失 | NA | S117 |
| jiangsuiyang | organ_T13_half_cl_20260908/short_checks_20260908 | — | 未记录 | 含背景分项缺失 | NA | S118 |
| jiangsuiyang | organ_T13_half_cl_20260908/subset | — | 未记录 | 含背景分项缺失 | NA | S119 |
| jiangsuiyang | organ_T13_half_cl_20260908/task_strategy_check/synthetic/run | 3 | 1 | 工程/短诊断；含背景分项缺失 | NA | S120 |
| jiangsuiyang | organ_T13_half_cl_20260908/task_strategy_check/synthetic_small_alpha/resume_t2 | 3 | 1 | 工程/短诊断；含背景分项缺失 | NA | S121 |
| jiangsuiyang | organ_T13_half_cl_20260908/task_strategy_check/synthetic_small_alpha/run | 3 | 1 | 工程/短诊断；含背景分项缺失 | NA | S122 |
| jiangsuiyang | organ_T2_coverage_20260908/public_release | — | 未记录 | 含背景分项缺失 | NA | S123 |
| jiangsuiyang | organ_T2_coverage_20260908/runs/e1_sw_lr01_g0 | 3 | 20 | 含背景分项缺失 | NA | S124 |
| jiangsuiyang | organ_T2_coverage_20260908/runs/e1_sw_lr01_g01 | 3 | 20 | 含背景分项缺失 | NA | S125 |
| jiangsuiyang | organ_T2_coverage_20260908/runs/e1_sw_lr03_g0 | 3 | 20 | 含背景分项缺失 | NA | S126 |
| jiangsuiyang | organ_T2_coverage_20260908/runs/e1_sw_lr03_g01 | 3 | 20 | 含背景分项缺失 | NA | S127 |
| jiangsuiyang | organ_T2_coverage_20260908/runs/ind_fg20 | 1 | 80 | 含背景分项缺失 | NA | S128 |
| jiangsuiyang | organ_T2_coverage_20260908/runs/ind_fg40 | 1 | 80 | 含背景分项缺失 | NA | S129 |
| jiangsuiyang | organ_T2_coverage_20260908/runs/stability_gate_mb4 | 2 | 2 | 含背景分项缺失 | NA | S130 |
| jiangsuiyang | organ_domain_swap_20260909/bidmc/t2 | 2 | 60 | 含背景分项缺失 | NA | S131 |
| jiangsuiyang | organ_domain_swap_20260909/bidmc/t3 | 3 | 60 | 含背景分项缺失 | NA | S132 |
| jiangsuiyang | organ_domain_swap_20260909/synthetic_check/resume_t2 | 3 | 2 | 工程/短诊断；含背景分项缺失 | NA | S133 |
| jiangsuiyang | organ_domain_swap_20260909/synthetic_check/run | 3 | 1 | 工程/短诊断；含背景分项缺失 | NA | S134 |
| jiangsuiyang | organ_domain_swap_20260909/ucl/t2 | 2 | 60 | 含背景分项缺失 | NA | S135 |
| jiangsuiyang | organ_domain_swap_20260909/ucl/t3 | 3 | 60 | 含背景分项缺失 | NA | S136 |
| jiangsuiyang | organ_t3_retention_probe_20260909/delivery | — | 未记录 | 含背景分项缺失 | NA | S137 |
| jiangsuiyang | tune_independent_A_07261_seed42_20260907_1303/oracle_pattern_f5_b10 | 1 | 150 | 工程/短诊断；含背景分项缺失 | NA | S138 |

### 7.1 早期参数搜索与 Joint 短诊断缺项

这些表保存的是前景指标，不能从其均值反推出背景分项；列出每个已有 run 和关键参数，含背景 Dice 留为 NA。

**zs_derpp_ab_sweep_20260902**（S139）

| run_id | global_weight | buffer_size | minibatch_size | alpha | beta | status | audit | 含背景 Dice | 指标来源 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| a2m7x | 1.0 | 64 | 8 | 0.25 | 0.5 | complete | PASS | NA | S139 |
| c4n9v | 1.0 | 64 | 8 | 0.5 | 0.5 | complete | PASS | NA | S139 |
| e6p3r | 1.0 | 64 | 8 | 0.5 | 1.0 | complete | PASS | NA | S139 |
| g8t5w | 1.0 | 64 | 8 | 1.0 | 1.0 | complete | PASS | NA | S139 |
| b3k8p | 0.1 | 64 | 8 | 1.0 | 1.0 | complete | PASS | NA | S139 |
| d5m2v | 1.0 | 128 | 8 | 1.0 | 1.0 | complete | PASS | NA | S139 |
| f7r4x | 1.0 | 64 | 16 | 1.0 | 1.0 | NA | NA | NA | S139 |
| h9t6z | 1.0 | 64 | 8 | 1.0 | 0.5 | complete | PASS | NA | S139 |

**organ_zs_derpp_ab_sweep_20260902**（S140）

| run_id | global_weight | buffer_size | minibatch_size | alpha | beta | audit | 含背景 Dice | 指标来源 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| j3p7n | 1.0 | 64 | 8 | 0.5 | 0.5 | PASS | NA | S140 |
| k5r9v | 1.0 | 64 | 8 | 1.0 | 0.5 | PASS | NA | S140 |
| l7t4x | 1.0 | 64 | 8 | 1.0 | 1.0 | PASS | NA | S140 |
| m2v6q | 1.0 | 64 | 8 | 0.25 | 0.5 | PASS | NA | S140 |
| n4x8r | 1.0 | 64 | 8 | 0.5 | 1.0 | PASS | NA | S140 |
| p6z3t | 0.1 | 64 | 8 | 0.5 | 0.5 | PASS | NA | S140 |

**joint_short_20260902**（S141）

| run_id | learning_rate | epochs | audit | 含背景 Dice | 指标来源 |
| --- | --- | --- | --- | --- | --- |
| h4m8q | .03 | 1 | PASS | 0.565419 | S142 |
| k7v2n | .06 | 1 | PASS | 0.561027 | S143 |
| p3x6d | .10 | 1 | PASS | 0.501509 | S144 |
| r8c4w | .02 | 3 | PASS | 0.677230 | S145 |
| t5n9b | .03 | 3 | PASS | 0.736325 | S146 |
| u2f7k | .04 | 3 | PASS | 0.765420 | S147 |
| w6d3s | .03 | 5 | PASS | 0.743595 | S148 |
| y9h4m | .04 | 5 | PASS | 0.776010 | 见第 4 节 |

Joint 的 y9h4m 已在第 4 节由归档背景复评文件补齐；其余 Joint 随本次补评更新，两类 replay 参数筛选仍缺背景分项。Domain 参数筛选中 f7r4x 的 minibatch=16 超出当时 24 GB 显存条件，属于不可行候选，不是完成的零分实验。

### 7.2 更早的静态参考、覆盖率与过拟合诊断

静态参考表含显式病例级含背景字段，保留如下。它是历史 checkpoint 导出结果，当前未重新确认其数据划分，不能并入正式测试主表。

| run | 变体 | 记录状态 | best checkpoint epoch（原始索引） | best 含背景 Dice | last 含背景 Dice | 来源 |
| --- | --- | --- | --- | --- | --- | --- |
| static_A0_sgd_seed42 | fg_only | completed | 4 | 0.146502 | 0.105977 | S149 |
| static_A_ratio_sgd_seed42 | legacy_ratio | completed | 10 | 0.509141 | 0.373760 | S149 |
| static_Dense_v2_sgd_seed42 | dense | completed | 136 | 0.818302 | 0.811501 | S149 |

| run | 记录任务 | 指标限制 | 含背景 Dice | 来源 |
| --- | --- | --- | --- | --- |
| coverage_runs/B1/pce_seed42_stage1 | 1 | 历史阶段指标；背景口径未确认 | NA | S150 |
| coverage_runs/B1/zs_seed42_stage1 | 1 | 历史阶段指标；背景口径未确认 | NA | S151 |
| coverage_runs/B2/pce_seed42_stage1 | 1 | 历史阶段指标；背景口径未确认 | NA | S152 |
| coverage_runs/B2/zs_seed42_stage1 | 1 | 历史阶段指标；背景口径未确认 | NA | S153 |
| coverage_runs/B3/pce_seed42_stage1 | 1 | 历史阶段指标；背景口径未确认 | NA | S154 |
| coverage_runs/B3/zs_seed42_stage1 | 1 | 历史阶段指标；背景口径未确认 | NA | S155 |
| coverage_runs/Dense/dense_seed42_stage1 | 1 | 历史阶段指标；背景口径未确认 | NA | S156 |
| overfit/pce_mib_seed42_stage2 | 1,2 | 历史阶段指标；背景口径未确认 | NA | S157 |
| overfit/pce_seed42_stage1 | 1 | 历史阶段指标；背景口径未确认 | NA | S158 |
| overfit/zs_mib_seed42_stage2 | 1,2 | 历史阶段指标；背景口径未确认 | NA | S159 |
| overfit/zs_seed42_stage1 | 1 | 历史阶段指标；背景口径未确认 | NA | S160 |
| overfit_fixed/pce_mib_seed42_stage2 | 1,2 | 历史阶段指标；背景口径未确认 | NA | S161 |

## 8. 无完整 summary 的产物索引

以下目录有 manifest，但未找到对应完整 summary。这里只登记记录状态，不把 manifest 里历史遗留的 `running` 解释为当前仍有训练进程。早期独立训练已由第 6 节 CSV 完整覆盖的目录、resume 前缀拷贝和参考镜像不重复列出。需要数值补齐时，应从保留 checkpoint 与同一评估协议复评；本次没有用猜测补值。

| 位置 | 批次 / run | 证据状态 | 配置预算 | 完整含背景 Dice | 来源 |
| --- | --- | --- | --- | --- | --- |
| my-gpu | class_comparisons_20260910/jobs/pce_sequential | 无最终 summary | 80 | NA | S162 |
| my-gpu | class_comparisons_20260910/jobs/zs_sequential | 无最终 summary | 80 | NA | S163 |
| my-gpu | class_er_gpu3_20260913/jobs/zs-er | 无最终 summary | 80 | NA | S164 |
| my-gpu | class_independent_spatial_sweep_20260908/jobs/cl_spatial_0.001 | 无最终 summary | 40 | NA | S165 |
| my-gpu | class_independent_spatial_sweep_20260908/jobs/ind_T1_formal | 无最终 summary | 80 | NA | S166 |
| my-gpu | class_independent_spatial_sweep_20260908/smoke_cl_spatial_original | 无最终 summary | 1 | NA | S167 |
| my-gpu | class_independent_spatial_sweep_20260908/smoke_ind_T3 | 无最终 summary | 1 | NA | S168 |
| my-gpu | class_replay_comparisons_20260912/jobs/zs-der | 无最终 summary | 80 | NA | S169 |
| my-gpu | class_replay_comparisons_20260912/jobs/zs-er | 无最终 summary | 80 | NA | S170 |
| my-gpu | class_replay_improve_20260915/jobs/zs-der-control5 | 无最终 summary | 80/5/60 | NA | S171 |
| my-gpu | class_replay_improve_20260915/jobs/zs-er-smoke | 无最终 summary | 80/1/60 | NA | S172 |
| my-gpu | class_replay_resume_20260913/jobs/zs-der | 无最终 summary | 80 | NA | S173 |
| my-gpu | class_replay_resume_20260913/jobs/zs-er | 无最终 summary | 80 | NA | S174 |
| my-gpu | core_runs/class_q8v2n6_seed42_150e_20260811T1345Z/zs_mib_raw_global1_diagnostic_stopped_iter1000_seed42 | 无最终 summary | 150 | NA | S175 |
| my-gpu | core_runs/class_q8v2n6_seed42_150e_20260811T1345Z/zs_sequential_raw_global1_diagnostic_stopped_iter1200_seed42 | 无最终 summary | 150 | NA | S176 |
| my-gpu | domain_joint_c3fix_20260902/runs/c3adam_e5_s42 | 历史 manifest 标 running；未确认存活 | 5 | NA | S177 |
| my-gpu | domain_joint_c3fix_20260902/runs/c3adamgd_e5_s42 | 历史 manifest 标 running；未确认存活 | 5 | NA | S178 |
| my-gpu | domain_joint_c3fix_20260902/runs/c3sgd_e3_s42 | 历史 manifest 标 running；未确认存活 | 3 | NA | S179 |
| my-gpu | domain_joint_validation_20260902/runs/balpce_b4e20_s42 | 历史 manifest 标 running；未确认存活 | 20 | NA | S180 |
| my-gpu | domain_joint_validation_20260902/runs/balzs_b4e20_s42 | 历史 manifest 标 running；未确认存活 | 20 | NA | S181 |
| my-gpu | organ_runs_20260901/runs/o1seq_r1 | 无最终 summary | 150 | NA | S182 |
| my-gpu | organ_runs_20260901/runs/o2ewc_r1 | 无最终 summary | 150 | NA | S183 |
| my-gpu | organ_runs_20260901/runs/u2k7m | 无最终 summary | 150 | NA | S184 |
| my-gpu | organ_runs_20260901/runs/u2k7m_r1 | 无最终 summary | 150 | NA | S185 |
| my-gpu | organ_runs_20260901/runs/u2k7m_r2 | 无最终 summary | 150 | NA | S186 |
| my-gpu | organ_runs_20260901/runs/v3p8n | 无最终 summary | 150 | NA | S187 |
| my-gpu | organ_runs_20260901/runs/v3p8n_r1 | 无最终 summary | 150 | NA | S188 |
| my-gpu | organ_runs_20260901/runs/v3p8n_r2 | 无最终 summary | 150 | NA | S189 |
| jiangsuiyang | class_independent_spatial_sweep_20260908/jobs/ind_T1_lr0.01 | 无最终 summary | 20 | NA | S190 |
| jiangsuiyang | class_independent_spatial_sweep_20260908/jobs/ind_T2_lr0.01 | 无最终 summary | 20 | NA | S191 |
| jiangsuiyang | class_independent_spatial_sweep_20260908/jobs/ind_T3_lr0.01 | 无最终 summary | 20 | NA | S192 |
| jiangsuiyang | class_independent_spatial_sweep_20260908/smoke_cl_spatial | 无最终 summary | 1 | NA | S193 |
| jiangsuiyang | independent80_seed42_20260906/smoke_class_T2_scribble | complete | 1 | NA | S194 |
| jiangsuiyang | independent80_seed42_20260906/smoke_domain_A_full | complete | 1 | NA | S195 |
| jiangsuiyang | organ_CL_throughput_20260908/spatial_off/run | 无最终 summary | 2 | NA | S196 |
| jiangsuiyang | organ_T13_half_cl_20260908/balance_checks_20260908_v2/balanced_current_full_replay/run | 无最终 summary | 60 | NA | S197 |
| jiangsuiyang | organ_T13_half_cl_20260908/balance_checks_20260908_v2/feature_replay_only/run | 无最终 summary | 60 | NA | S198 |
| jiangsuiyang | organ_T13_half_cl_20260908/balance_checks_20260908_v2/no_replay_losses/run | 无最终 summary | 60 | NA | S199 |
| jiangsuiyang | organ_T13_half_cl_20260908/balance_checks_20260908_v2/supervision_replay_only/run | 无最终 summary | 60 | NA | S200 |
| jiangsuiyang | organ_T13_half_cl_20260908/diagnostic_t2_seed43_v3/run | 无最终 summary | 60 | NA | S201 |
| jiangsuiyang | organ_T13_half_cl_20260908/formal_controls_20260908/alpha0 | 无最终 summary | 60 | NA | S202 |
| jiangsuiyang | organ_T13_half_cl_20260908/formal_controls_20260908/alpha01 | 无最终 summary | 60 | NA | S203 |
| jiangsuiyang | organ_T13_half_cl_20260908/formal_controls_20260908/no_spatial | 无最终 summary | 60 | NA | S204 |
| jiangsuiyang | organ_T13_half_cl_20260908/run | 无最终 summary | 80 | NA | S205 |
| jiangsuiyang | organ_T13_half_cl_20260908/run60 | 无最终 summary | 60 | NA | S206 |
| jiangsuiyang | organ_T13_half_cl_20260908/short_checks_20260908/clean_bn/run | 无最终 summary | 60 | NA | S207 |
| jiangsuiyang | organ_T13_half_cl_20260908/short_checks_20260908/clean_bn_clip5_126/run | 无最终 summary | 60 | NA | S208 |
| jiangsuiyang | organ_T13_half_cl_20260908/short_checks_20260908/clip5/run | 无最终 summary | 60 | NA | S209 |
| jiangsuiyang | organ_T13_half_cl_20260908/short_checks_20260908/clip5_126/run | 无最终 summary | 60 | NA | S210 |
| jiangsuiyang | organ_T13_half_cl_20260908/short_checks_20260908/clip5_420/run | 无最终 summary | 60 | NA | S211 |
| jiangsuiyang | organ_T13_half_cl_20260908/short_checks_20260908/clip5_seed44_126/run | 无最终 summary | 60 | NA | S212 |
| jiangsuiyang | organ_T13_half_cl_20260908/short_checks_20260908/frozen_bn_126/run | 无最终 summary | 60 | NA | S213 |
| jiangsuiyang | organ_T13_half_cl_20260908/short_checks_20260908/frozen_bn_clip5_126/run | 无最终 summary | 60 | NA | S214 |
| jiangsuiyang | organ_T13_half_cl_20260908/short_checks_20260908/lr003/run | 无最终 summary | 60 | NA | S215 |
| jiangsuiyang | organ_T13_half_cl_20260908/short_checks_20260908/lr003_126/run | 无最终 summary | 60 | NA | S216 |
| jiangsuiyang | organ_T13_half_cl_20260908/short_checks_20260908/lr003_clip5/run | 无最终 summary | 60 | NA | S217 |
| jiangsuiyang | organ_T13_half_cl_20260908/short_checks_20260908/lr003_clip5_126/run | 无最终 summary | 60 | NA | S218 |
| jiangsuiyang | organ_T13_half_cl_20260908/small_alpha_checks_20260908/alpha_0.01/run | 无最终 summary | 60 | NA | S219 |
| jiangsuiyang | organ_T13_half_cl_20260908/small_alpha_checks_20260908/alpha_0.05/run | 无最终 summary | 60 | NA | S220 |
| jiangsuiyang | organ_T13_half_cl_20260908/small_alpha_checks_20260908/alpha_0.1/run | 无最终 summary | 60 | NA | S221 |
| jiangsuiyang | organ_T2_coverage_20260908/runs/cl_fg20 | 无最终 summary | 80 | NA | S222 |
| jiangsuiyang | organ_T2_coverage_20260908/runs/cl_fg40 | 无最终 summary | 80 | NA | S223 |
| jiangsuiyang | organ_T2_coverage_20260908/runs/stability_gate | 无最终 summary | 2 | NA | S224 |
| jiangsuiyang | organ_T2_coverage_20260908/runs/sw_lr01_g0 | 无最终 summary | 20 | NA | S225 |
| jiangsuiyang | organ_T2_coverage_20260908/runs/sw_lr01_g01 | 无最终 summary | 20 | NA | S226 |
| jiangsuiyang | organ_T2_coverage_20260908/runs/sw_lr03_g0 | 无最终 summary | 20 | NA | S227 |
| jiangsuiyang | organ_T2_coverage_20260908/runs/sw_lr03_g01 | 无最终 summary | 20 | NA | S228 |
| jiangsuiyang | organ_T34_lr006_spatial_pair20_20260910/runs/spatial0 | 无最终 summary | 20 | NA | S229 |
| jiangsuiyang | organ_T34_lr006_spatial_pair20_20260910/runs/spatial001 | 无最终 summary | 20 | NA | S230 |
| jiangsuiyang | organ_batch8_probe_20260908/b8_spatial_on/run | 无最终 summary | 1 | NA | S231 |
| jiangsuiyang | organ_metrics_half_20260909/prefix_T2 | 无最终 summary | 60 | NA | S232 |
| jiangsuiyang | organ_metrics_half_20260909/runs/ind_T4 | 无最终 summary | 80 | NA | S233 |
| jiangsuiyang | organ_t3_retention_probe_20260909/R0 | 无最终 summary | 60 | NA | S234 |
| jiangsuiyang | organ_t3_retention_probe_20260909/R1 | 无最终 summary | 60 | NA | S235 |
| jiangsuiyang | organ_t3_retention_probe_20260909/R2 | 无最终 summary | 60 | NA | S236 |
| jiangsuiyang | organ_t3_retention_probe_20260909/R3 | 无最终 summary | 60 | NA | S237 |
| jiangsuiyang | organ_t3_retention_probe_20260909/frozen | 无最终 summary | 60 | NA | S238 |
| jiangsuiyang | replay_comparisons_20260911/runs/domain_zs-der | 无最终 summary | 80 | NA | S239 |
| jiangsuiyang | replay_comparisons_20260911/runs/domain_zs-er | 无最终 summary | 80 | NA | S240 |
| jiangsuiyang | replay_comparisons_20260911/runs/organ_zs-der | 无最终 summary | 40 | NA | S241 |
| jiangsuiyang | tune_independent_A_07261_seed42_20260907_1303/independent | stopped_protocol_mismatch | 150 | NA | S242 |
| jiangsuiyang | tune_independent_A_07261_seed42_20260907_1303/oracle | stopped_protocol_mismatch | 150 | NA | S243 |
| jiangsuiyang | tune_independent_A_07261_seed42_20260907_1303/parity/independent | 历史 manifest 标 running；未确认存活 | 150 | NA | S244 |
| jiangsuiyang | tune_independent_A_07261_seed42_20260907_1303/parity/oracle | 历史 manifest 标 running；未确认存活 | 150 | NA | S245 |
| jiangsuiyang | tune_independent_A_07261_seed42_20260907_1303/parity_deterministic/independent | 历史 manifest 标 running；未确认存活 | 150 | NA | S246 |
| jiangsuiyang | tune_independent_A_07261_seed42_20260907_1303/parity_deterministic/oracle | 历史 manifest 标 running；未确认存活 | 150 | NA | S247 |
| jiangsuiyang | tune_independent_A_07261_seed42_20260907_1303/parity_pattern_f5_b10/independent | 历史 manifest 标 running；未确认存活 | 150 | NA | S248 |
| jiangsuiyang | tune_independent_A_07261_seed42_20260907_1303/parity_pattern_f5_b10/oracle | 历史 manifest 标 running；未确认存活 | 150 | NA | S249 |
| jiangsuiyang | tune_independent_A_07261_seed42_20260907_1303/parity_pattern_f5_b10/oracle_repeat | 历史 manifest 标 running；未确认存活 | 150 | NA | S250 |
| jiangsuiyang | tune_independent_A_07261_seed42_20260907_1303/parity_repeat/independent | 历史 manifest 标 running；未确认存活 | 150 | NA | S251 |
| jiangsuiyang | tune_independent_A_07261_seed42_20260907_1303/parity_repeat/oracle | 历史 manifest 标 running；未确认存活 | 150 | NA | S252 |
| jiangsuiyang | tune_independent_A_07261_seed42_20260907_1303/parity_repeat/oracle_repeat | 历史 manifest 标 running；未确认存活 | 150 | NA | S253 |
| jiangsuiyang | zs_independent_search_seed42_20260906_inclusive_aborted_20260906_2128/tuning/class_T2_c01 | 历史 manifest 标 running；未确认存活 | 20 | NA | S254 |
| jiangsuiyang | zs_independent_search_seed42_20260906_inclusive_aborted_20260906_2128/tuning/class_T2_c02 | 历史 manifest 标 running；未确认存活 | 20 | NA | S255 |
| jiangsuiyang | zs_independent_search_seed42_20260906_inclusive_aborted_20260906_2128/tuning/class_T2_c03 | 历史 manifest 标 running；未确认存活 | 20 | NA | S256 |
| jiangsuiyang | zs_independent_search_seed42_20260906_inclusive_aborted_20260906_2128/tuning/class_T2_c04 | 历史 manifest 标 running；未确认存活 | 20 | NA | S257 |
| jiangsuiyang | zs_independent_search_seed42_20260906_inclusive_aborted_20260906_2128/tuning/domain_C_c01 | 历史 manifest 标 running；未确认存活 | 20 | NA | S258 |
| jiangsuiyang | zs_independent_search_seed42_20260906_inclusive_aborted_20260906_2128/tuning/domain_C_c02 | 历史 manifest 标 running；未确认存活 | 20 | NA | S259 |
| jiangsuiyang | zs_independent_search_seed42_20260906_inclusive_aborted_20260906_2128/tuning/domain_C_c03 | 历史 manifest 标 running；未确认存活 | 20 | NA | S260 |
| jiangsuiyang | zs_independent_search_seed42_20260906_inclusive_aborted_20260906_2128/tuning/domain_C_c04 | 历史 manifest 标 running；未确认存活 | 20 | NA | S261 |
| jiangsuiyang | zs_independent_tuning_seed42_20260906/smoke_class_T2 | complete | 2 | NA | S262 |
| jiangsuiyang | zs_independent_tuning_seed42_20260906/smoke_domain_B_skiptest | complete | 1 | NA | S263 |

## 8A. Checkpoint 检索与本次补评

本节为后续补充。测试和验证沿用各自原始划分，使用同一选中 checkpoint；不改选模、不重训。加载时检查大小及修改时间稳定，严格匹配模型权重；同时重算前景 Dice，要求与原记录的最大绝对差不超过 0.0001。没有原前景依据或不匹配的值保留待核查，不写入正式表。已有 runtime 支持时使用其 HDF5 内存缓存，以减少 NAS 小块读取；不改变预测及指标算法。

配套保存了[补评标量与路径占位配置](../results/background_recovery_20260915/aggregate.json)和[补评脚本](../reevaluate_background.py)。原始权重、数据及训练日志保留在服务器。执行时将脚本以中性名称 run.py 放入私有运行目录，并提供恢复实际路径后的 plan.json；不得把公开占位路径直接视为可执行配置。

本次快照已有 **35 个补评通过一致性检查**。后台队列可在快照后继续产生结果；以下状态不是永久实时状态。

| 服务器 | 批次 | 计划评估数 | 完成且一致 | 状态 | 快照时间 |
| --- | --- | --- | --- | --- | --- |
| my-gpu | background_recovery_20260915 | 13 | 13 | complete | 2026-09-15T10:45:47.777305+00:00 |
| my-gpu | background_recovery_20260915/extra | 4 | 4 | complete | 2026-09-15T10:45:47.777305+00:00 |
| jiangsuiyang | background_recovery_20260915 | 28 | 0 | running | 2026-09-15T18:45:47.878669+08:00 |
| jiangsuiyang | background_recovery_20260915/legacy | 17 | 17 | complete | 2026-09-15T18:45:47.878669+08:00 |
| jiangsuiyang | background_recovery_20260915/cached | 1 | 1 | complete | 2026-09-15T10:50:56.771032+00:00 |

| 服务器 | 原 run | 阶段 | 划分 | 确切 checkpoint | 补评状态 | 含背景均值 | 补评来源 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| my-gpu | class_independent_spatial_sweep_20260908/jobs/ind_T1_lr0.01 | 1 | val | `class_independent_spatial_sweep_20260908/jobs/ind_T1_lr0.01/s01.pt` | complete | 0.735565 | S089 |
| my-gpu | class_independent_spatial_sweep_20260908/jobs/ind_T1_lr0.03 | 1 | val | `class_independent_spatial_sweep_20260908/jobs/ind_T1_lr0.03/s01.pt` | complete | 0.692540 | S090 |
| my-gpu | class_independent_spatial_sweep_20260908/jobs/ind_T2_lr0.01 | 1 | val | `class_independent_spatial_sweep_20260908/jobs/ind_T2_lr0.01/s01.pt` | complete | 0.599318 | S091 |
| my-gpu | class_independent_spatial_sweep_20260908/jobs/ind_T2_lr0.03 | 1 | val | `class_independent_spatial_sweep_20260908/jobs/ind_T2_lr0.03/s01.pt` | complete | 0.658331 | S092 |
| my-gpu | class_independent_spatial_sweep_20260908/jobs/ind_T3_lr0.01 | 1 | val | `class_independent_spatial_sweep_20260908/jobs/ind_T3_lr0.01/s01.pt` | complete | 0.711128 | S093 |
| my-gpu | class_independent_spatial_sweep_20260908/jobs/ind_T3_lr0.03 | 1 | val | `class_independent_spatial_sweep_20260908/jobs/ind_T3_lr0.03/s01.pt` | complete | 0.713457 | S094 |
| my-gpu | core_runs/class_q8v2n6_seed42_150e_20260811T1345Z/pce_mib_seed42 | 3 | test | `core_runs/class_q8v2n6_seed42_150e_20260811T1345Z/pce_mib_seed42/s03.pt` | complete | 0.666446 | S100 |
| my-gpu | core_runs/class_q8v2n6_seed42_150e_20260811T1345Z/pce_sequential_seed42 | 3 | test | `core_runs/class_q8v2n6_seed42_150e_20260811T1345Z/pce_sequential_seed42/s03.pt` | complete | 0.423118 | S101 |
| my-gpu | core_runs/class_q8v2n6_seed42_150e_20260811T1345Z/zs_mib_global1_origscale_seed42 | 3 | test | `core_runs/class_q8v2n6_seed42_150e_20260811T1345Z/zs_mib_global1_origscale_seed42/s03.pt` | complete | 0.805777 | S102 |
| my-gpu | core_runs/class_q8v2n6_seed42_150e_20260811T1345Z/zs_sequential_global1_origscale_seed42 | 3 | test | `core_runs/class_q8v2n6_seed42_150e_20260811T1345Z/zs_sequential_global1_origscale_seed42/s03.pt` | complete | 0.461417 | S103 |
| my-gpu | class_independent_spatial_sweep_20260908/jobs/cl_spatial_0.01 | 3 | val | `class_independent_spatial_sweep_20260908/jobs/cl_spatial_0.01/s03.pt` | complete | 0.775111 | S088 |
| my-gpu | domain_runs_20260901/runs/d0gpm_r1 | 6 | test | `domain_runs_20260901/runs/d0gpm_r1/s06.pt` | complete | 0.469249 | S106 |
| my-gpu | domain_runs_20260901/runs/d3p9n_r1 | 6 | test | `domain_runs_20260901/runs/d3p9n_r1/s06.pt` | complete | 0.451765 | S107 |
| my-gpu | class_replay_improve_20260915_r2/jobs/zs-der-control5 | 2 | val | `class_replay_improve_20260915_r2/jobs/zs-der-control5/s02.pt` | complete | 0.378416 | S050 |
| my-gpu | class_replay_improve_20260915_r2/jobs/zs-der-improved5 | 2 | val | `class_replay_improve_20260915_r2/jobs/zs-der-improved5/s02.pt` | complete | 0.361179 | S051 |
| my-gpu | class_replay_improve_20260915_r2/jobs/zs-er-control5 | 2 | val | `class_replay_improve_20260915_r2/jobs/zs-er-control5/s02.pt` | complete | 0.404902 | S048 |
| my-gpu | class_replay_improve_20260915_r2/jobs/zs-er-improved5 | 2 | val | `class_replay_improve_20260915_r2/jobs/zs-er-improved5/s02.pt` | complete | 0.696645 | S049 |
| jiangsuiyang | organ_T4_from_T3best_lr006_spatial_pair10_20260910/runs/spatial0 | 4 | test | `organ_T4_from_T3best_lr006_spatial_pair10_20260910/runs/spatial0/s04.pt` | 已定位；后台待评 | NA | S264 |
| jiangsuiyang | organ_T4_from_T3best_lr006_spatial_pair10_20260910/runs/spatial001 | 4 | test | `organ_T4_from_T3best_lr006_spatial_pair10_20260910/runs/spatial001/s04.pt` | 已定位；后台待评 | NA | S265 |
| jiangsuiyang | organ_comparisons_20260910/runs/dense-sequential | 4 | test | `organ_comparisons_20260910/runs/dense-sequential/s04.pt` | 已定位；后台待评 | NA | S266 |
| jiangsuiyang | organ_comparisons_20260910/runs/pce-sequential | 4 | test | `organ_comparisons_20260910/runs/pce-sequential/s04.pt` | 已定位；后台待评 | NA | S267 |
| jiangsuiyang | organ_comparisons_20260910/runs/zs-ewc | 4 | test | `organ_comparisons_20260910/runs/zs-ewc/s04.pt` | 已定位；后台待评 | NA | S268 |
| jiangsuiyang | organ_comparisons_20260910/runs/zs-gpm | 4 | test | `organ_comparisons_20260910/runs/zs-gpm/s04.pt` | 已定位；后台待评 | NA | S269 |
| jiangsuiyang | organ_comparisons_20260910/runs/zs-sequential | 4 | test | `organ_comparisons_20260910/runs/zs-sequential/s04.pt` | 已定位；后台待评 | NA | S270 |
| jiangsuiyang | organ_baseline_tuning_20260911/runs/zs-ewc_formal | 4 | test | `organ_baseline_tuning_20260911/runs/zs-ewc_formal/s04.pt` | 已定位；后台待评 | NA | S271 |
| jiangsuiyang | organ_baseline_tuning_20260911/runs/zs-gpm_formal | 4 | test | `organ_baseline_tuning_20260911/runs/zs-gpm_formal/s04.pt` | 已定位；后台待评 | NA | S272 |
| jiangsuiyang | organ_baseline_tuning_20260911/runs/zs-sequential_formal | 4 | test | `organ_baseline_tuning_20260911/runs/zs-sequential_formal/s04.pt` | 已定位；后台待评 | NA | S273 |
| jiangsuiyang | organ_metrics_T134_half_20260909/runs/ind_T4 | 1 | test | `organ_metrics_T134_half_20260909/runs/ind_T4/s01.pt` | 已定位；后台待评 | NA | S274 |
| jiangsuiyang | organ_metrics_half_20260909/runs/ind_T3 | 1 | test | `organ_metrics_half_20260909/runs/ind_T3/s01.pt` | 已定位；后台待评 | NA | S275 |
| jiangsuiyang | organ_T3_lr006_spatial_pair10_20260910/runs/spatial0 | 3 | test | `organ_T3_lr006_spatial_pair10_20260910/runs/spatial0/s03.pt` | 已定位；后台待评 | NA | S276 |
| jiangsuiyang | organ_T3_lr006_spatial_pair10_20260910/runs/spatial001 | 3 | test | `organ_T3_lr006_spatial_pair10_20260910/runs/spatial001/s03.pt` | 已定位；后台待评 | NA | S277 |
| jiangsuiyang | organ_baseline_tuning_20260911/runs/zs-ewc_c0 | 4 | val | `organ_baseline_tuning_20260911/runs/zs-ewc_c0/s04.pt` | 已定位；后台待评 | NA | S278 |
| jiangsuiyang | organ_baseline_tuning_20260911/runs/zs-ewc_c1 | 4 | val | `organ_baseline_tuning_20260911/runs/zs-ewc_c1/s04.pt` | 已定位；后台待评 | NA | S279 |
| jiangsuiyang | organ_baseline_tuning_20260911/runs/zs-ewc_c2 | 4 | val | `organ_baseline_tuning_20260911/runs/zs-ewc_c2/s04.pt` | 已定位；后台待评 | NA | S280 |
| jiangsuiyang | organ_baseline_tuning_20260911/runs/zs-ewc_c3 | 4 | val | `organ_baseline_tuning_20260911/runs/zs-ewc_c3/s04.pt` | 已定位；后台待评 | NA | S281 |
| jiangsuiyang | organ_baseline_tuning_20260911/runs/zs-gpm_c0 | 4 | val | `organ_baseline_tuning_20260911/runs/zs-gpm_c0/s04.pt` | 已定位；后台待评 | NA | S282 |
| jiangsuiyang | organ_baseline_tuning_20260911/runs/zs-gpm_c1 | 4 | val | `organ_baseline_tuning_20260911/runs/zs-gpm_c1/s04.pt` | 已定位；后台待评 | NA | S283 |
| jiangsuiyang | organ_baseline_tuning_20260911/runs/zs-gpm_c2 | 4 | val | `organ_baseline_tuning_20260911/runs/zs-gpm_c2/s04.pt` | 已定位；后台待评 | NA | S284 |
| jiangsuiyang | organ_baseline_tuning_20260911/runs/zs-gpm_c3 | 4 | val | `organ_baseline_tuning_20260911/runs/zs-gpm_c3/s04.pt` | 已定位；后台待评 | NA | S285 |
| jiangsuiyang | organ_baseline_tuning_20260911/runs/zs-sequential_c0 | 4 | val | `organ_baseline_tuning_20260911/runs/zs-sequential_c0/s04.pt` | 已定位；后台待评 | NA | S286 |
| jiangsuiyang | organ_baseline_tuning_20260911/runs/zs-sequential_c1 | 4 | val | `organ_baseline_tuning_20260911/runs/zs-sequential_c1/s04.pt` | 已定位；后台待评 | NA | S287 |
| jiangsuiyang | organ_baseline_tuning_20260911/runs/zs-sequential_c2 | 4 | val | `organ_baseline_tuning_20260911/runs/zs-sequential_c2/s04.pt` | 已定位；后台待评 | NA | S288 |
| jiangsuiyang | organ_baseline_tuning_20260911/runs/zs-sequential_c3 | 4 | val | `organ_baseline_tuning_20260911/runs/zs-sequential_c3/s04.pt` | 已定位；后台待评 | NA | S289 |
| jiangsuiyang | organ_metrics_half_20260909/runs/organ_no_replay | 4 | test | `organ_metrics_half_20260909/runs/organ_no_replay/s04.pt` | 已定位；后台待评 | NA | S290 |
| jiangsuiyang | organ_metrics_half_20260909/runs/organ_retention | 4 | test | `organ_metrics_half_20260909/runs/organ_retention/s04.pt` | 已定位；后台待评 | NA | S291 |
| jiangsuiyang | legacy_joint/h4m8q | 1 | test | `<legacy q1d7f>/runs/h4m8q/s01.pt` | complete | 0.565419 | S142 |
| jiangsuiyang | legacy_joint/k7v2n | 1 | test | `<legacy q1d7f>/runs/k7v2n/s01.pt` | complete | 0.561027 | S143 |
| jiangsuiyang | legacy_joint/p3x6d | 1 | test | `<legacy q1d7f>/runs/p3x6d/s01.pt` | complete | 0.501509 | S144 |
| jiangsuiyang | legacy_joint/r8c4w | 1 | test | `<legacy q1d7f>/runs/r8c4w/s01.pt` | complete | 0.677230 | S145 |
| jiangsuiyang | legacy_joint/t5n9b | 1 | test | `<legacy q1d7f>/runs/t5n9b/s01.pt` | complete | 0.736325 | S146 |
| jiangsuiyang | legacy_joint/u2f7k | 1 | test | `<legacy q1d7f>/runs/u2f7k/s01.pt` | complete | 0.765420 | S147 |
| jiangsuiyang | legacy_joint/w6d3s | 1 | test | `<legacy q1d7f>/runs/w6d3s/s01.pt` | complete | 0.743595 | S148 |
| jiangsuiyang | legacy_domain/m7v2q | 1 | test | `<legacy q1d7f>/runs/m7v2q/s01.pt` | complete | 0.860545 | S292 |
| jiangsuiyang | legacy_domain/m7v2q | 2 | test | `<legacy q1d7f>/runs/m7v2q/s02.pt` | complete | 0.768055 | S293 |
| jiangsuiyang | legacy_domain/m7v2q | 3 | test | `<legacy q1d7f>/runs/m7v2q/s03.pt` | complete | 0.844347 | S294 |
| jiangsuiyang | legacy_domain/m7v2q | 4 | test | `<legacy q1d7f>/runs/m7v2q/s04.pt` | complete | 0.778141 | S295 |
| jiangsuiyang | legacy_domain/m7v2q | 5 | test | `<legacy q1d7f>/runs/m7v2q/s05.pt` | complete | 0.866099 | S296 |
| jiangsuiyang | legacy_domain/t4m7b | 1 | test | `<legacy NAS runs>/t4m7b/s01.pt` | complete | 0.770834 | S297 |
| jiangsuiyang | legacy_domain/t4m7b | 2 | test | `<legacy NAS runs>/t4m7b/s02.pt` | complete | 0.844488 | S298 |
| jiangsuiyang | legacy_domain/t4m7b | 3 | test | `<legacy NAS runs>/t4m7b/s03.pt` | complete | 0.847956 | S299 |
| jiangsuiyang | legacy_domain/t4m7b | 4 | test | `<legacy NAS runs>/t4m7b/s04.pt` | complete | 0.844937 | S300 |
| jiangsuiyang | legacy_domain/t4m7b | 5 | test | `<legacy NAS runs>/t4m7b/s05.pt` | complete | 0.871841 | S301 |
| jiangsuiyang | organ_T4_from_T3best_lr006_spatial_pair10_20260910/runs/spatial0 | 4 | test | `organ_T4_from_T3best_lr006_spatial_pair10_20260910/runs/spatial0/s04.pt` | complete | 0.848059 | S015 |

上述 legacy_domain 记录仅计算首次学完对应任务的对角线值，用于后续含背景 BWTR；其单任务均值不等于六域最终 A-Dice。标为“后台待评”的来源为预定输出位置，尚非已有结果证据。

### 已完成补评的每任务结果

| 服务器 | run | 阶段 | 划分 | 任务 | 背景 Dice | 含背景 Dice | 前景重放最大差 | 来源 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| my-gpu | class_independent_spatial_sweep_20260908/jobs/ind_T1_lr0.01 | 1 | val | T1 | 0.980145 | 0.735565 | 0.000000 | S089 |
| my-gpu | class_independent_spatial_sweep_20260908/jobs/ind_T1_lr0.03 | 1 | val | T1 | 0.970673 | 0.692540 | 0.000000 | S090 |
| my-gpu | class_independent_spatial_sweep_20260908/jobs/ind_T2_lr0.01 | 1 | val | T2 | 0.982733 | 0.599318 | 0.000000 | S091 |
| my-gpu | class_independent_spatial_sweep_20260908/jobs/ind_T2_lr0.03 | 1 | val | T2 | 0.983541 | 0.658331 | 0.000000 | S092 |
| my-gpu | class_independent_spatial_sweep_20260908/jobs/ind_T3_lr0.01 | 1 | val | T3 | 0.998224 | 0.711128 | 0.000000 | S093 |
| my-gpu | class_independent_spatial_sweep_20260908/jobs/ind_T3_lr0.03 | 1 | val | T3 | 0.998325 | 0.713457 | 0.000000 | S094 |
| my-gpu | core_runs/class_q8v2n6_seed42_150e_20260811T1345Z/pce_mib_seed42 | 3 | test | T1 | 0.931045 | 0.675441 | 0.000000 | S100 |
| my-gpu | core_runs/class_q8v2n6_seed42_150e_20260811T1345Z/pce_mib_seed42 | 3 | test | T2 | 0.924686 | 0.688768 | 0.000000 | S100 |
| my-gpu | core_runs/class_q8v2n6_seed42_150e_20260811T1345Z/pce_mib_seed42 | 3 | test | T3 | 0.916251 | 0.635130 | 0.000000 | S100 |
| my-gpu | core_runs/class_q8v2n6_seed42_150e_20260811T1345Z/pce_sequential_seed42 | 3 | test | T1 | 0.967222 | 0.241805 | 0.000000 | S101 |
| my-gpu | core_runs/class_q8v2n6_seed42_150e_20260811T1345Z/pce_sequential_seed42 | 3 | test | T2 | 0.973908 | 0.324636 | 0.000000 | S101 |
| my-gpu | core_runs/class_q8v2n6_seed42_150e_20260811T1345Z/pce_sequential_seed42 | 3 | test | T3 | 0.991817 | 0.702913 | 0.000000 | S101 |
| my-gpu | core_runs/class_q8v2n6_seed42_150e_20260811T1345Z/zs_mib_global1_origscale_seed42 | 3 | test | T1 | 0.969613 | 0.822527 | 0.000000 | S102 |
| my-gpu | core_runs/class_q8v2n6_seed42_150e_20260811T1345Z/zs_mib_global1_origscale_seed42 | 3 | test | T2 | 0.962291 | 0.827738 | 0.000000 | S102 |
| my-gpu | core_runs/class_q8v2n6_seed42_150e_20260811T1345Z/zs_mib_global1_origscale_seed42 | 3 | test | T3 | 0.952174 | 0.767067 | 0.000000 | S102 |
| my-gpu | core_runs/class_q8v2n6_seed42_150e_20260811T1345Z/zs_sequential_global1_origscale_seed42 | 3 | test | T1 | 0.970692 | 0.242673 | 0.000000 | S103 |
| my-gpu | core_runs/class_q8v2n6_seed42_150e_20260811T1345Z/zs_sequential_global1_origscale_seed42 | 3 | test | T2 | 0.977135 | 0.325712 | 0.000000 | S103 |
| my-gpu | core_runs/class_q8v2n6_seed42_150e_20260811T1345Z/zs_sequential_global1_origscale_seed42 | 3 | test | T3 | 0.996968 | 0.815867 | 0.000000 | S103 |
| my-gpu | class_independent_spatial_sweep_20260908/jobs/cl_spatial_0.01 | 3 | val | T1 | 0.974339 | 0.830800 | 0.000000 | S088 |
| my-gpu | class_independent_spatial_sweep_20260908/jobs/cl_spatial_0.01 | 3 | val | T2 | 0.970031 | 0.719103 | 0.000000 | S088 |
| my-gpu | class_independent_spatial_sweep_20260908/jobs/cl_spatial_0.01 | 3 | val | T3 | 0.966795 | 0.775431 | 0.000000 | S088 |
| my-gpu | domain_runs_20260901/runs/d0gpm_r1 | 6 | test | A | 0.809257 | 0.450111 | 0.000000 | S106 |
| my-gpu | domain_runs_20260901/runs/d0gpm_r1 | 6 | test | B | 0.827789 | 0.461325 | 0.000000 | S106 |
| my-gpu | domain_runs_20260901/runs/d0gpm_r1 | 6 | test | C | 0.836460 | 0.489359 | 0.000000 | S106 |
| my-gpu | domain_runs_20260901/runs/d0gpm_r1 | 6 | test | D | 0.843736 | 0.465462 | 0.000000 | S106 |
| my-gpu | domain_runs_20260901/runs/d0gpm_r1 | 6 | test | E | 0.854268 | 0.507230 | 0.000000 | S106 |
| my-gpu | domain_runs_20260901/runs/d0gpm_r1 | 6 | test | F | 0.825791 | 0.442008 | 0.000000 | S106 |
| my-gpu | domain_runs_20260901/runs/d3p9n_r1 | 6 | test | A | 0.806092 | 0.447455 | 0.000000 | S107 |
| my-gpu | domain_runs_20260901/runs/d3p9n_r1 | 6 | test | B | 0.807468 | 0.447668 | 0.000000 | S107 |
| my-gpu | domain_runs_20260901/runs/d3p9n_r1 | 6 | test | C | 0.813152 | 0.470232 | 0.000000 | S107 |
| my-gpu | domain_runs_20260901/runs/d3p9n_r1 | 6 | test | D | 0.809657 | 0.442087 | 0.000000 | S107 |
| my-gpu | domain_runs_20260901/runs/d3p9n_r1 | 6 | test | E | 0.811379 | 0.473352 | 0.000000 | S107 |
| my-gpu | domain_runs_20260901/runs/d3p9n_r1 | 6 | test | F | 0.806425 | 0.429799 | 0.000000 | S107 |
| my-gpu | class_replay_improve_20260915_r2/jobs/zs-der-control5 | 2 | val | T1 | 0.970863 | 0.242716 | 0.000000 | S050 |
| my-gpu | class_replay_improve_20260915_r2/jobs/zs-der-control5 | 2 | val | T2 | 0.980762 | 0.514116 | 0.000000 | S050 |
| my-gpu | class_replay_improve_20260915_r2/jobs/zs-der-improved5 | 2 | val | T1 | 0.960327 | 0.240082 | 0.000000 | S051 |
| my-gpu | class_replay_improve_20260915_r2/jobs/zs-der-improved5 | 2 | val | T2 | 0.970961 | 0.482276 | 0.000000 | S051 |
| my-gpu | class_replay_improve_20260915_r2/jobs/zs-er-control5 | 2 | val | T1 | 0.975094 | 0.243774 | 0.000000 | S048 |
| my-gpu | class_replay_improve_20260915_r2/jobs/zs-er-control5 | 2 | val | T2 | 0.982763 | 0.566031 | 0.000000 | S048 |
| my-gpu | class_replay_improve_20260915_r2/jobs/zs-er-improved5 | 2 | val | T1 | 0.973292 | 0.716085 | 0.000000 | S049 |
| my-gpu | class_replay_improve_20260915_r2/jobs/zs-er-improved5 | 2 | val | T2 | 0.969062 | 0.677205 | 0.000000 | S049 |
| jiangsuiyang | legacy_joint/h4m8q | 1 | test | A | 0.985951 | 0.571555 | 0.000000 | S142 |
| jiangsuiyang | legacy_joint/h4m8q | 1 | test | B | 0.992055 | 0.564861 | 0.000000 | S142 |
| jiangsuiyang | legacy_joint/h4m8q | 1 | test | C | 0.990207 | 0.667044 | 0.000000 | S142 |
| jiangsuiyang | legacy_joint/h4m8q | 1 | test | D | 0.991978 | 0.516848 | 0.000000 | S142 |
| jiangsuiyang | legacy_joint/h4m8q | 1 | test | E | 0.986748 | 0.526301 | 0.000000 | S142 |
| jiangsuiyang | legacy_joint/h4m8q | 1 | test | F | 0.993549 | 0.545905 | 0.000000 | S142 |
| jiangsuiyang | legacy_joint/k7v2n | 1 | test | A | 0.982701 | 0.534363 | 0.000000 | S143 |
| jiangsuiyang | legacy_joint/k7v2n | 1 | test | B | 0.992319 | 0.707325 | 0.000000 | S143 |
| jiangsuiyang | legacy_joint/k7v2n | 1 | test | C | 0.987451 | 0.592917 | 0.000000 | S143 |
| jiangsuiyang | legacy_joint/k7v2n | 1 | test | D | 0.988552 | 0.504852 | 0.000000 | S143 |
| jiangsuiyang | legacy_joint/k7v2n | 1 | test | E | 0.983141 | 0.496573 | 0.000000 | S143 |
| jiangsuiyang | legacy_joint/k7v2n | 1 | test | F | 0.990314 | 0.530132 | 0.000000 | S143 |
| jiangsuiyang | legacy_joint/p3x6d | 1 | test | A | 0.980647 | 0.491838 | 0.000000 | S144 |
| jiangsuiyang | legacy_joint/p3x6d | 1 | test | B | 0.990783 | 0.498853 | 0.000000 | S144 |
| jiangsuiyang | legacy_joint/p3x6d | 1 | test | C | 0.988614 | 0.523446 | 0.000000 | S144 |
| jiangsuiyang | legacy_joint/p3x6d | 1 | test | D | 0.991916 | 0.497930 | 0.000000 | S144 |
| jiangsuiyang | legacy_joint/p3x6d | 1 | test | E | 0.987475 | 0.499205 | 0.000000 | S144 |
| jiangsuiyang | legacy_joint/p3x6d | 1 | test | F | 0.994276 | 0.497785 | 0.000000 | S144 |
| jiangsuiyang | legacy_joint/r8c4w | 1 | test | A | 0.965944 | 0.634316 | 0.000000 | S145 |
| jiangsuiyang | legacy_joint/r8c4w | 1 | test | B | 0.983453 | 0.719403 | 0.000000 | S145 |
| jiangsuiyang | legacy_joint/r8c4w | 1 | test | C | 0.985462 | 0.741127 | 0.000000 | S145 |
| jiangsuiyang | legacy_joint/r8c4w | 1 | test | D | 0.990312 | 0.612023 | 0.000000 | S145 |
| jiangsuiyang | legacy_joint/r8c4w | 1 | test | E | 0.986936 | 0.730748 | 0.000000 | S145 |
| jiangsuiyang | legacy_joint/r8c4w | 1 | test | F | 0.976533 | 0.625761 | 0.000000 | S145 |
| jiangsuiyang | legacy_joint/t5n9b | 1 | test | A | 0.981636 | 0.696957 | 0.000000 | S146 |
| jiangsuiyang | legacy_joint/t5n9b | 1 | test | B | 0.986345 | 0.743327 | 0.000000 | S146 |
| jiangsuiyang | legacy_joint/t5n9b | 1 | test | C | 0.987805 | 0.778989 | 0.000000 | S146 |
| jiangsuiyang | legacy_joint/t5n9b | 1 | test | D | 0.993916 | 0.777883 | 0.000000 | S146 |
| jiangsuiyang | legacy_joint/t5n9b | 1 | test | E | 0.987062 | 0.750752 | 0.000000 | S146 |
| jiangsuiyang | legacy_joint/t5n9b | 1 | test | F | 0.984085 | 0.670040 | 0.000000 | S146 |
| jiangsuiyang | legacy_joint/u2f7k | 1 | test | A | 0.982719 | 0.714278 | 0.000000 | S147 |
| jiangsuiyang | legacy_joint/u2f7k | 1 | test | B | 0.990721 | 0.790918 | 0.000000 | S147 |
| jiangsuiyang | legacy_joint/u2f7k | 1 | test | C | 0.990882 | 0.802399 | 0.000000 | S147 |
| jiangsuiyang | legacy_joint/u2f7k | 1 | test | D | 0.995276 | 0.804098 | 0.000000 | S147 |
| jiangsuiyang | legacy_joint/u2f7k | 1 | test | E | 0.988250 | 0.768150 | 0.000000 | S147 |
| jiangsuiyang | legacy_joint/u2f7k | 1 | test | F | 0.988882 | 0.712678 | 0.000000 | S147 |
| jiangsuiyang | legacy_joint/w6d3s | 1 | test | A | 0.983350 | 0.723432 | 0.000000 | S148 |
| jiangsuiyang | legacy_joint/w6d3s | 1 | test | B | 0.988191 | 0.762867 | 0.000000 | S148 |
| jiangsuiyang | legacy_joint/w6d3s | 1 | test | C | 0.988166 | 0.780244 | 0.000000 | S148 |
| jiangsuiyang | legacy_joint/w6d3s | 1 | test | D | 0.993989 | 0.766292 | 0.000000 | S148 |
| jiangsuiyang | legacy_joint/w6d3s | 1 | test | E | 0.990754 | 0.785984 | 0.000000 | S148 |
| jiangsuiyang | legacy_joint/w6d3s | 1 | test | F | 0.979889 | 0.642752 | 0.000000 | S148 |
| jiangsuiyang | legacy_domain/m7v2q | 1 | test | A | 0.994949 | 0.860545 | 0.000000 | S292 |
| jiangsuiyang | legacy_domain/m7v2q | 2 | test | B | 0.988053 | 0.768055 | 0.000000 | S293 |
| jiangsuiyang | legacy_domain/m7v2q | 3 | test | C | 0.991456 | 0.844347 | 0.000000 | S294 |
| jiangsuiyang | legacy_domain/m7v2q | 4 | test | D | 0.992568 | 0.778141 | 0.000000 | S295 |
| jiangsuiyang | legacy_domain/m7v2q | 5 | test | E | 0.993446 | 0.866099 | 0.000000 | S296 |
| jiangsuiyang | legacy_domain/t4m7b | 1 | test | A | 0.989414 | 0.770834 | 0.000000 | S297 |
| jiangsuiyang | legacy_domain/t4m7b | 2 | test | B | 0.994052 | 0.844488 | 0.000000 | S298 |
| jiangsuiyang | legacy_domain/t4m7b | 3 | test | C | 0.992320 | 0.847956 | 0.000000 | S299 |
| jiangsuiyang | legacy_domain/t4m7b | 4 | test | D | 0.996225 | 0.844937 | 0.000000 | S300 |
| jiangsuiyang | legacy_domain/t4m7b | 5 | test | E | 0.994344 | 0.871841 | 0.000000 | S301 |
| jiangsuiyang | organ_T4_from_T3best_lr006_spatial_pair10_20260910/runs/spatial0 | 4 | test | T1 | 0.994827 | 0.814544 | 0.000000 | S015 |
| jiangsuiyang | organ_T4_from_T3best_lr006_spatial_pair10_20260910/runs/spatial0 | 4 | test | T2 | 0.995778 | 0.784317 | 0.000000 | S015 |
| jiangsuiyang | organ_T4_from_T3best_lr006_spatial_pair10_20260910/runs/spatial0 | 4 | test | T3 | 0.982703 | 0.887446 | 0.000000 | S015 |
| jiangsuiyang | organ_T4_from_T3best_lr006_spatial_pair10_20260910/runs/spatial0 | 4 | test | T4 | 0.992339 | 0.905929 | 0.000000 | S015 |

### 其余条目的 checkpoint 可用性

下表登记原报告相关来源目录。sNN.pt 是完成阶段保存的选中权重；只有 best/state 而没有完成 summary 的记录只能作 partial。大型 state 文件不逐一列出。仅定位到文件不代表已经通过加载和数据一致性校验。

| 服务器 | 目录 | 可定位权重 | 限制 |
| --- | --- | --- | --- |
| my-gpu | class_comparisons_20260910_neutral_restart/jobs/pce_sequential | `s01.pt`, `s02.pt`, `s02_best.pt`, `s03.pt`, `s03_best.pt`, `s01_best.pt` | 存在阶段权重；待按原协议匹配 |
| my-gpu | class_comparisons_20260910_neutral_restart/jobs/dense_sequential | `s01.pt`, `s02.pt`, `s02_best.pt`, `s03.pt`, `s03_best.pt`, `s01_best.pt` | 存在阶段权重；待按原协议匹配 |
| my-gpu | class_comparisons_20260910_neutral_restart/jobs/zs_sequential | `s01.pt`, `s02.pt`, `s02_best.pt`, `s03.pt`, `s03_best.pt`, `s01_best.pt` | 存在阶段权重；待按原协议匹配 |
| my-gpu | class_comparisons_20260910_neutral_restart/jobs/zs_ewc | `s01.pt`, `s02.pt`, `s02_best.pt`, `s03.pt`, `s03_best.pt`, `s01_best.pt` | 存在阶段权重；待按原协议匹配 |
| my-gpu | class_comparisons_20260910_neutral_restart/jobs/zs_gpm | `s01.pt`, `s02.pt`, `s02_best.pt`, `s03.pt`, `s03_best.pt`, `s01_best.pt` | 存在阶段权重；待按原协议匹配 |
| my-gpu | class_comparisons_20260910_neutral_restart/jobs/main_s0 | `s01.pt`, `s02.pt`, `s02_best.pt`, `s03.pt`, `s03_best.pt`, `s01_best.pt` | 存在阶段权重；待按原协议匹配 |
| my-gpu | class_replay_60e_20260913/jobs/zs-er | `s01.pt`, `s02.pt`, `s02_best.pt`, `s03.pt`, `s03_best.pt`, `s01_best.pt` | 存在阶段权重；待按原协议匹配 |
| my-gpu | class_replay_60e_20260913/jobs/zs-der | `s01.pt`, `s02.pt`, `s02_best.pt`, `s03.pt`, `s03_best.pt`, `s01_best.pt` | 存在阶段权重；待按原协议匹配 |
| my-gpu | class_comparisons_20260910_neutral_restart/jobs/ind_T2_s0 | `s01.pt`, `s01_best.pt` | 存在阶段权重；待按原协议匹配 |
| my-gpu | class_comparisons_20260910_neutral_restart/jobs/ind_T3_s0 | `s01.pt`, `s01_best.pt` | 存在阶段权重；待按原协议匹配 |
| my-gpu | class_comparisons_20260910/dense_smoke | `s01.pt`, `s01_best.pt` | 存在阶段权重；待按原协议匹配 |
| my-gpu | class_comparisons_20260910/gpm_smoke | `s01.pt`, `s02.pt`, `s02_best.pt`, `s01_best.pt` | 存在阶段权重；待按原协议匹配 |
| my-gpu | class_independent_spatial_sweep_20260908/smoke_cl_spatial_original_gco | `s01.pt`, `s02.pt`, `s02_best.pt`, `s03.pt`, `s03_best.pt`, `s01_best.pt` | 存在阶段权重；待按原协议匹配 |
| my-gpu | class_independent_spatial_sweep_20260908/smoke_ind_T3_gco | `s01.pt`, `s01_best.pt` | 存在阶段权重；待按原协议匹配 |
| my-gpu | class_replay_improve_20260915/jobs/zs-der-smoke | `s01.pt`, `s02.pt`, `s02_best.pt`, `s01_best.pt` | 存在阶段权重；待按原协议匹配 |
| my-gpu | class_replay_improve_20260915_r2/jobs/zs-der-smoke | `s01.pt`, `s02.pt`, `s02_best.pt`, `s01_best.pt` | 存在阶段权重；待按原协议匹配 |
| my-gpu | class_replay_improve_20260915_r2/jobs/zs-er-smoke | `s01.pt`, `s02.pt`, `s02_best.pt`, `s01_best.pt` | 存在阶段权重；待按原协议匹配 |
| my-gpu | domain_joint_runs_20260901/runs/j0b2l15_r1 | `joint_best.pt`, `joint_model.pt` | 仅 best/state 或诊断权重；最终状态未确认 |
| my-gpu | domain_joint_runs_20260901/runs/j1b4l30_r1 | `joint_best.pt`, `joint_model.pt` | 仅 best/state 或诊断权重；最终状态未确认 |
| my-gpu | domain_runs_20260901/smoke/joint_smoke_b8 | `joint_best.pt`, `joint_model.pt` | 仅 best/state 或诊断权重；最终状态未确认 |
| my-gpu | scribblecl_domain_organ_20260809/runs/domain/domain_pce_ft_seed42_20260809T082951Z | — | 本次目标目录顶层未检索到 checkpoint |
| my-gpu | scribblecl_domain_organ_20260809/runs/organ/organ_pce_ft_seed42_20260809T083415Z | — | 本次目标目录顶层未检索到 checkpoint |
| my-gpu | class_comparisons_20260910/jobs/pce_sequential | `s01.pt`, `s02_best.pt`, `s01_best.pt` | 存在阶段权重；待按原协议匹配 |
| my-gpu | class_comparisons_20260910/jobs/zs_sequential | `s01_best.pt` | 仅 best/state 或诊断权重；最终状态未确认 |
| my-gpu | class_er_gpu3_20260913/jobs/zs-er | `s01.pt`, `s02.pt`, `s02_best.pt`, `s03_best.pt`, `s01_best.pt` | 存在阶段权重；待按原协议匹配 |
| my-gpu | class_independent_spatial_sweep_20260908/jobs/cl_spatial_0.001 | — | 本次目标目录顶层未检索到 checkpoint |
| my-gpu | class_independent_spatial_sweep_20260908/jobs/ind_T1_formal | `s01_best.pt` | 仅 best/state 或诊断权重；最终状态未确认 |
| my-gpu | class_independent_spatial_sweep_20260908/smoke_cl_spatial_original | — | 本次目标目录顶层未检索到 checkpoint |
| my-gpu | class_independent_spatial_sweep_20260908/smoke_ind_T3 | — | 本次目标目录顶层未检索到 checkpoint |
| my-gpu | class_replay_comparisons_20260912/jobs/zs-der | `s01_best.pt` | 仅 best/state 或诊断权重；最终状态未确认 |
| my-gpu | class_replay_comparisons_20260912/jobs/zs-er | `s01.pt`, `s02_best.pt`, `s01_best.pt` | 存在阶段权重；待按原协议匹配 |
| my-gpu | class_replay_improve_20260915/jobs/zs-der-control5 | `s01.pt`, `s01_best.pt` | 存在阶段权重；待按原协议匹配 |
| my-gpu | class_replay_improve_20260915/jobs/zs-er-smoke | `s01.pt`, `s01_best.pt` | 存在阶段权重；待按原协议匹配 |
| my-gpu | class_replay_resume_20260913/jobs/zs-der | `s01.pt`, `s01_best.pt` | 存在阶段权重；待按原协议匹配 |
| my-gpu | class_replay_resume_20260913/jobs/zs-er | `s01.pt`, `s02_best.pt`, `s01_best.pt` | 存在阶段权重；待按原协议匹配 |
| my-gpu | core_runs/class_q8v2n6_seed42_150e_20260811T1345Z/zs_mib_raw_global1_diagnostic_stopped_iter1000_seed42 | `s01_best.pt` | 仅 best/state 或诊断权重；最终状态未确认 |
| my-gpu | core_runs/class_q8v2n6_seed42_150e_20260811T1345Z/zs_sequential_raw_global1_diagnostic_stopped_iter1200_seed42 | `s01_best.pt` | 仅 best/state 或诊断权重；最终状态未确认 |
| my-gpu | domain_joint_c3fix_20260902/runs/c3adam_e5_s42 | `joint_best.pt` | 仅 best/state 或诊断权重；最终状态未确认 |
| my-gpu | domain_joint_c3fix_20260902/runs/c3adamgd_e5_s42 | `joint_best.pt` | 仅 best/state 或诊断权重；最终状态未确认 |
| my-gpu | domain_joint_c3fix_20260902/runs/c3sgd_e3_s42 | `joint_best.pt` | 仅 best/state 或诊断权重；最终状态未确认 |
| my-gpu | domain_joint_validation_20260902/runs/balpce_b4e20_s42 | `joint_best.pt` | 仅 best/state 或诊断权重；最终状态未确认 |
| my-gpu | domain_joint_validation_20260902/runs/balzs_b4e20_s42 | `joint_best.pt` | 仅 best/state 或诊断权重；最终状态未确认 |
| my-gpu | organ_runs_20260901/runs/o1seq_r1 | `s01_best.pt` | 仅 best/state 或诊断权重；最终状态未确认 |
| my-gpu | organ_runs_20260901/runs/o2ewc_r1 | `s01_best.pt` | 仅 best/state 或诊断权重；最终状态未确认 |
| my-gpu | organ_runs_20260901/runs/u2k7m | — | 本次目标目录顶层未检索到 checkpoint |
| my-gpu | organ_runs_20260901/runs/u2k7m_r1 | `s01_best.pt` | 仅 best/state 或诊断权重；最终状态未确认 |
| my-gpu | organ_runs_20260901/runs/u2k7m_r2 | `s01_best.pt` | 仅 best/state 或诊断权重；最终状态未确认 |
| my-gpu | organ_runs_20260901/runs/v3p8n | — | 本次目标目录顶层未检索到 checkpoint |
| my-gpu | organ_runs_20260901/runs/v3p8n_r1 | — | 本次目标目录顶层未检索到 checkpoint |
| my-gpu | organ_runs_20260901/runs/v3p8n_r2 | — | 本次目标目录顶层未检索到 checkpoint |
| jiangsuiyang | replay_comparisons_20260911/runs/organ_zs-er | `s01_best.pt`, `s01.pt`, `s02_best.pt`, `s02.pt`, `s03_best.pt`, `s03.pt`, `s04_best.pt`, `s04.pt` | 存在阶段权重；待按原协议匹配 |
| jiangsuiyang | organ_comparisons_20260910/runs/dense-sequential | `s01_best.pt`, `s01.pt`, `s02_best.pt`, `s02.pt`, `s03_best.pt`, `s03.pt`, `s04_best.pt`, `s04.pt` | 存在阶段权重；待按原协议匹配 |
| jiangsuiyang | organ_comparisons_20260910/runs/pce-sequential | `s01_best.pt`, `s01.pt`, `s02_best.pt`, `s02.pt`, `s03_best.pt`, `s03.pt`, `s04_best.pt`, `s04.pt` | 存在阶段权重；待按原协议匹配 |
| jiangsuiyang | organ_comparisons_20260910/runs/zs-sequential | `s01_best.pt`, `s01.pt`, `s02_best.pt`, `s02.pt`, `s03_best.pt`, `s03.pt`, `s04_best.pt`, `s04.pt` | 存在阶段权重；待按原协议匹配 |
| jiangsuiyang | organ_comparisons_20260910/runs/zs-ewc | `s01_best.pt`, `s01.pt`, `s02_best.pt`, `s02.pt`, `s03_best.pt`, `s03.pt`, `s04_best.pt`, `s04.pt` | 存在阶段权重；待按原协议匹配 |
| jiangsuiyang | organ_comparisons_20260910/runs/zs-gpm | `s01_best.pt`, `s01.pt`, `s02_best.pt`, `s02.pt`, `s03_best.pt`, `s03.pt`, `s04_best.pt`, `s04.pt` | 存在阶段权重；待按原协议匹配 |
| jiangsuiyang | organ_baseline_tuning_20260911/runs/zs-sequential_formal | `s01_best.pt`, `s01.pt`, `s02_best.pt`, `s02.pt`, `s03_best.pt`, `s03.pt`, `s04_best.pt`, `s04.pt` | 存在阶段权重；待按原协议匹配 |
| jiangsuiyang | organ_baseline_tuning_20260911/runs/zs-ewc_formal | `s01_best.pt`, `s01.pt`, `s02_best.pt`, `s02.pt`, `s03_best.pt`, `s03.pt`, `s04_best.pt`, `s04.pt` | 存在阶段权重；待按原协议匹配 |
| jiangsuiyang | organ_baseline_tuning_20260911/runs/zs-gpm_formal | `s01_best.pt`, `s01.pt`, `s02_best.pt`, `s02.pt`, `s03_best.pt`, `s03.pt`, `s04_best.pt`, `s04.pt` | 存在阶段权重；待按原协议匹配 |
| jiangsuiyang | organ_T3_lr006_spatial_pair10_20260910/runs/spatial0 | `s01.pt`, `s02.pt`, `s03_best.pt`, `t3_epoch_01.pt`, `t3_epoch_02.pt`, `t3_epoch_03.pt`, `t3_epoch_04.pt`, `t3_epoch_05.pt`, `t3_epoch_06.pt`, `t3_epoch_07.pt`, `t3_epoch_08.pt`, `t3_epoch_09.pt`, `t3_epoch_10.pt`, `s03.pt` | 存在阶段权重；待按原协议匹配 |
| jiangsuiyang | organ_T3_lr006_spatial_pair10_20260910/runs/spatial001 | `s01.pt`, `s02.pt`, `s03_best.pt`, `t3_epoch_01.pt`, `t3_epoch_02.pt`, `t3_epoch_03.pt`, `t3_epoch_04.pt`, `t3_epoch_05.pt`, `t3_epoch_06.pt`, `t3_epoch_07.pt`, `t3_epoch_08.pt`, `t3_epoch_09.pt`, `t3_epoch_10.pt`, `s03.pt` | 存在阶段权重；待按原协议匹配 |
| jiangsuiyang | organ_T4_from_T3best_lr006_spatial_pair10_20260910/runs/spatial001 | `s01.pt`, `s02.pt`, `s03.pt`, `s04_best.pt`, `s04.pt` | 存在阶段权重；待按原协议匹配 |
| jiangsuiyang | organ_baseline_tuning_20260911/runs/zs-ewc_c0 | `s01_best.pt`, `s01.pt`, `s02_best.pt`, `s02.pt`, `s03_best.pt`, `s03.pt`, `s04_best.pt`, `s04.pt` | 存在阶段权重；待按原协议匹配 |
| jiangsuiyang | organ_baseline_tuning_20260911/runs/zs-ewc_c1 | `s01_best.pt`, `s01.pt`, `s02_best.pt`, `s02.pt`, `s03_best.pt`, `s03.pt`, `s04_best.pt`, `s04.pt` | 存在阶段权重；待按原协议匹配 |
| jiangsuiyang | organ_baseline_tuning_20260911/runs/zs-ewc_c2 | `s01_best.pt`, `s01.pt`, `s02_best.pt`, `s02.pt`, `s03_best.pt`, `s03.pt`, `s04_best.pt`, `s04.pt` | 存在阶段权重；待按原协议匹配 |
| jiangsuiyang | organ_baseline_tuning_20260911/runs/zs-ewc_c3 | `s01_best.pt`, `s01.pt`, `s02_best.pt`, `s02.pt`, `s03_best.pt`, `s03.pt`, `s04_best.pt`, `s04.pt` | 存在阶段权重；待按原协议匹配 |
| jiangsuiyang | organ_baseline_tuning_20260911/runs/zs-gpm_c0 | `s01_best.pt`, `s01.pt`, `s02_best.pt`, `s02.pt`, `s03_best.pt`, `s03.pt`, `s04_best.pt`, `s04.pt` | 存在阶段权重；待按原协议匹配 |
| jiangsuiyang | organ_baseline_tuning_20260911/runs/zs-gpm_c1 | `s01_best.pt`, `s01.pt`, `s02_best.pt`, `s02.pt`, `s03_best.pt`, `s03.pt`, `s04_best.pt`, `s04.pt` | 存在阶段权重；待按原协议匹配 |
| jiangsuiyang | organ_baseline_tuning_20260911/runs/zs-gpm_c2 | `s01_best.pt`, `s01.pt`, `s02_best.pt`, `s02.pt`, `s03_best.pt`, `s03.pt`, `s04_best.pt`, `s04.pt` | 存在阶段权重；待按原协议匹配 |
| jiangsuiyang | organ_baseline_tuning_20260911/runs/zs-gpm_c3 | `s01_best.pt`, `s01.pt`, `s02_best.pt`, `s02.pt`, `s03_best.pt`, `s03.pt`, `s04_best.pt`, `s04.pt` | 存在阶段权重；待按原协议匹配 |
| jiangsuiyang | organ_baseline_tuning_20260911/runs/zs-sequential_c0 | `s01_best.pt`, `s01.pt`, `s02_best.pt`, `s02.pt`, `s03_best.pt`, `s03.pt`, `s04_best.pt`, `s04.pt` | 存在阶段权重；待按原协议匹配 |
| jiangsuiyang | organ_baseline_tuning_20260911/runs/zs-sequential_c1 | `s01_best.pt`, `s01.pt`, `s02_best.pt`, `s02.pt`, `s03_best.pt`, `s03.pt`, `s04_best.pt`, `s04.pt` | 存在阶段权重；待按原协议匹配 |
| jiangsuiyang | organ_baseline_tuning_20260911/runs/zs-sequential_c2 | `s01_best.pt`, `s01.pt`, `s02_best.pt`, `s02.pt`, `s03_best.pt`, `s03.pt`, `s04_best.pt`, `s04.pt` | 存在阶段权重；待按原协议匹配 |
| jiangsuiyang | organ_baseline_tuning_20260911/runs/zs-sequential_c3 | `s01_best.pt`, `s01.pt`, `s02_best.pt`, `s02.pt`, `s03_best.pt`, `s03.pt`, `s04_best.pt`, `s04.pt` | 存在阶段权重；待按原协议匹配 |
| jiangsuiyang | organ_metrics_T134_half_20260909/runs/ind_T4 | `s01_best.pt`, `s01.pt` | 存在阶段权重；待按原协议匹配 |
| jiangsuiyang | organ_metrics_half_20260909/runs/ind_T3 | `s01_best.pt`, `s01.pt` | 存在阶段权重；待按原协议匹配 |
| jiangsuiyang | organ_metrics_half_20260909/runs/organ_no_replay | `s01.pt`, `s02_best.pt`, `s02.pt`, `s03_best.pt`, `s03.pt`, `s04_best.pt`, `s04.pt` | 存在阶段权重；待按原协议匹配 |
| jiangsuiyang | organ_metrics_half_20260909/runs/organ_retention | `s01.pt`, `s02.pt`, `s03_best.pt`, `s03.pt`, `s04_best.pt`, `s04.pt` | 存在阶段权重；待按原协议匹配 |
| jiangsuiyang | independent_A_spatial_sweep20_formal80_20260907/formal80 | `best.pt`, `last.pt` | 仅 best/state 或诊断权重；最终状态未确认 |
| jiangsuiyang | independent_A_spatial_sweep20_formal80_20260907/sweep_s00 | `best.pt`, `last.pt` | 仅 best/state 或诊断权重；最终状态未确认 |
| jiangsuiyang | independent_A_spatial_sweep20_formal80_20260907/sweep_s01 | `best.pt`, `last.pt` | 仅 best/state 或诊断权重；最终状态未确认 |
| jiangsuiyang | independent_A_spatial_sweep20_formal80_20260907/sweep_s02 | `best.pt`, `last.pt` | 仅 best/state 或诊断权重；最终状态未确认 |
| jiangsuiyang | independent_A_spatial_sweep20_formal80_20260907/sweep_s03 | `best.pt`, `last.pt` | 仅 best/state 或诊断权重；最终状态未确认 |
| jiangsuiyang | independent_A_spatial_sweep20_formal80_20260907/sweep_s04 | `best.pt`, `last.pt` | 仅 best/state 或诊断权重；最终状态未确认 |
| jiangsuiyang | independent_A_spatial_sweep20_formal80_20260907/sweep_s05 | `best.pt`, `last.pt` | 仅 best/state 或诊断权重；最终状态未确认 |
| jiangsuiyang | independent_A_warmup_sweep80_20260907/warmup10 | `best.pt`, `last.pt` | 仅 best/state 或诊断权重；最终状态未确认 |
| jiangsuiyang | independent_A_warmup_sweep80_20260907/warmup20 | `best.pt`, `last.pt` | 仅 best/state 或诊断权重；最终状态未确认 |
| jiangsuiyang | independent_A_warmup_sweep80_20260907/warmup40 | `best.pt`, `last.pt` | 仅 best/state 或诊断权重；最终状态未确认 |
| jiangsuiyang | independent_A_warmup_sweep80_20260907/warmup60 | `best.pt`, `last.pt` | 仅 best/state 或诊断权重；最终状态未确认 |
| jiangsuiyang | independent_BF_formal80_warmup10_20260907/B | `best.pt`, `last.pt` | 仅 best/state 或诊断权重；最终状态未确认 |
| jiangsuiyang | independent_BF_formal80_warmup10_20260907/C | `best.pt`, `last.pt` | 仅 best/state 或诊断权重；最终状态未确认 |
| jiangsuiyang | independent_BF_formal80_warmup10_20260907/D | `best.pt`, `last.pt` | 仅 best/state 或诊断权重；最终状态未确认 |
| jiangsuiyang | independent_BF_formal80_warmup10_20260907/E | `best.pt`, `last.pt` | 仅 best/state 或诊断权重；最终状态未确认 |
| jiangsuiyang | independent_BF_formal80_warmup10_20260907/F | `best.pt`, `last.pt` | 仅 best/state 或诊断权重；最终状态未确认 |
| jiangsuiyang | independent_B_shared_smoke_20260907 | `best.pt`, `last.pt` | 仅 best/state 或诊断权重；最终状态未确认 |
| jiangsuiyang | independent_demo_smoke_20260907 | `best.pt`, `last.pt` | 仅 best/state 或诊断权重；最终状态未确认 |
| jiangsuiyang | organ_T2_original_scribble_20260908/runs/organ_fg20 | `best.pt`, `last.pt` | 仅 best/state 或诊断权重；最终状态未确认 |
| jiangsuiyang | organ_T2_reference_recovery_20260908/runs/domain_control | `best.pt`, `last.pt` | 仅 best/state 或诊断权重；最终状态未确认 |
| jiangsuiyang | organ_T2_reference_recovery_20260908/runs/organ_reference | `best.pt`, `last.pt` | 仅 best/state 或诊断权重；最终状态未确认 |
| jiangsuiyang | organ_T2_spatial30_deterministic_20260908/runs/domain_control | `best.pt`, `last.pt` | 仅 best/state 或诊断权重；最终状态未确认 |
| jiangsuiyang | organ_T2_spatial30_deterministic_20260908/runs/organ_reference | `best.pt`, `last.pt` | 仅 best/state 或诊断权重；最终状态未确认 |
| jiangsuiyang | organ_T2_spatial_start30_20260908/runs/domain_control | `best.pt`, `last.pt` | 仅 best/state 或诊断权重；最终状态未确认 |
| jiangsuiyang | organ_T2_spatial_start30_20260908/runs/organ_reference | `best.pt`, `last.pt` | 仅 best/state 或诊断权重；最终状态未确认 |
| jiangsuiyang | spatial_sweep_smoke_20260907 | `best.pt`, `last.pt` | 仅 best/state 或诊断权重；最终状态未确认 |
| jiangsuiyang | tune_independent_A_07261_seed42_20260907_1303/independent_pattern_f5_b10 | `best.pt`, `last.pt` | 仅 best/state 或诊断权重；最终状态未确认 |
| jiangsuiyang | tune_independent_A_07261_seed42_20260907_1303/smoke | `best.pt`, `last.pt` | 仅 best/state 或诊断权重；最终状态未确认 |
| jiangsuiyang | class_independent_spatial_sweep_20260908/smoke_cl_spatial_chunked | `s01_best.pt`, `s01.pt`, `s02_best.pt`, `s02.pt`, `s03_best.pt`, `s03.pt` | 存在阶段权重；待按原协议匹配 |
| jiangsuiyang | class_independent_spatial_sweep_20260908/smoke_ind_T3 | `s01_best.pt`, `s01.pt` | 存在阶段权重；待按原协议匹配 |
| jiangsuiyang | organ_CL_throughput_20260908/memory_spatial_off/run | — | 本次目标目录顶层未检索到 checkpoint |
| jiangsuiyang | organ_CL_throughput_20260908/memory_spatial_on/run | — | 本次目标目录顶层未检索到 checkpoint |
| jiangsuiyang | organ_T13_half_cl_20260908/balance_checks_20260908_v2 | — | 本次目标目录顶层未检索到 checkpoint |
| jiangsuiyang | organ_T13_half_cl_20260908/diagnostic_t2_seed43_v3 | — | 本次目标目录顶层未检索到 checkpoint |
| jiangsuiyang | organ_T13_half_cl_20260908/formal_small_alpha_20260908 | `s02_best.pt`, `s02.pt`, `s03_best.pt`, `s03.pt` | 存在阶段权重；待按原协议匹配 |
| jiangsuiyang | organ_T13_half_cl_20260908/short_checks_20260908 | — | 本次目标目录顶层未检索到 checkpoint |
| jiangsuiyang | organ_T13_half_cl_20260908/subset | — | 本次目标目录顶层未检索到 checkpoint |
| jiangsuiyang | organ_T13_half_cl_20260908/task_strategy_check/synthetic/run | `s01_best.pt`, `s01.pt`, `s02_best.pt`, `s02.pt`, `s03_best.pt`, `s03.pt` | 存在阶段权重；待按原协议匹配 |
| jiangsuiyang | organ_T13_half_cl_20260908/task_strategy_check/synthetic_small_alpha/resume_t2 | `s02_best.pt`, `s02.pt`, `s03_best.pt`, `s03.pt` | 存在阶段权重；待按原协议匹配 |
| jiangsuiyang | organ_T13_half_cl_20260908/task_strategy_check/synthetic_small_alpha/run | `s01_best.pt`, `s01.pt`, `s02_best.pt`, `s02.pt`, `s03_best.pt`, `s03.pt` | 存在阶段权重；待按原协议匹配 |
| jiangsuiyang | organ_T2_coverage_20260908/public_release | — | 本次目标目录顶层未检索到 checkpoint |
| jiangsuiyang | organ_T2_coverage_20260908/runs/e1_sw_lr01_g0 | `s03_best.pt`, `s03.pt` | 存在阶段权重；待按原协议匹配 |
| jiangsuiyang | organ_T2_coverage_20260908/runs/e1_sw_lr01_g01 | `s03_best.pt`, `s03.pt` | 存在阶段权重；待按原协议匹配 |
| jiangsuiyang | organ_T2_coverage_20260908/runs/e1_sw_lr03_g0 | `s03_best.pt`, `s03.pt` | 存在阶段权重；待按原协议匹配 |
| jiangsuiyang | organ_T2_coverage_20260908/runs/e1_sw_lr03_g01 | `s03_best.pt`, `s03.pt` | 存在阶段权重；待按原协议匹配 |
| jiangsuiyang | organ_T2_coverage_20260908/runs/ind_fg20 | `s01_best.pt`, `s01.pt` | 存在阶段权重；待按原协议匹配 |
| jiangsuiyang | organ_T2_coverage_20260908/runs/ind_fg40 | `s01_best.pt`, `s01.pt` | 存在阶段权重；待按原协议匹配 |
| jiangsuiyang | organ_T2_coverage_20260908/runs/stability_gate_mb4 | `s01_best.pt`, `s01.pt`, `s02_best.pt`, `s02.pt` | 存在阶段权重；待按原协议匹配 |
| jiangsuiyang | organ_domain_swap_20260909/bidmc/t2 | `s02_best.pt`, `s02.pt` | 存在阶段权重；待按原协议匹配 |
| jiangsuiyang | organ_domain_swap_20260909/bidmc/t3 | `s03_best.pt`, `s03.pt` | 存在阶段权重；待按原协议匹配 |
| jiangsuiyang | organ_domain_swap_20260909/synthetic_check/resume_t2 | `s02_best.pt`, `s02.pt`, `s03_best.pt`, `s03.pt` | 存在阶段权重；待按原协议匹配 |
| jiangsuiyang | organ_domain_swap_20260909/synthetic_check/run | `s01_best.pt`, `s01.pt`, `s02_best.pt`, `s02.pt`, `s03_best.pt`, `s03.pt` | 存在阶段权重；待按原协议匹配 |
| jiangsuiyang | organ_domain_swap_20260909/ucl/t2 | `s02_best.pt`, `s02.pt` | 存在阶段权重；待按原协议匹配 |
| jiangsuiyang | organ_domain_swap_20260909/ucl/t3 | `s03_best.pt`, `s03.pt` | 存在阶段权重；待按原协议匹配 |
| jiangsuiyang | organ_t3_retention_probe_20260909/delivery | — | 本次目标目录顶层未检索到 checkpoint |
| jiangsuiyang | tune_independent_A_07261_seed42_20260907_1303/oracle_pattern_f5_b10 | `s01_best.pt`, `s01.pt` | 存在阶段权重；待按原协议匹配 |
| jiangsuiyang | class_independent_spatial_sweep_20260908/jobs/ind_T1_lr0.01 | — | 本次目标目录顶层未检索到 checkpoint |
| jiangsuiyang | class_independent_spatial_sweep_20260908/jobs/ind_T2_lr0.01 | — | 本次目标目录顶层未检索到 checkpoint |
| jiangsuiyang | class_independent_spatial_sweep_20260908/jobs/ind_T3_lr0.01 | — | 本次目标目录顶层未检索到 checkpoint |
| jiangsuiyang | class_independent_spatial_sweep_20260908/smoke_cl_spatial | `s01_best.pt`, `s01.pt`, `s02_best.pt`, `s02.pt` | 存在阶段权重；待按原协议匹配 |
| jiangsuiyang | independent80_seed42_20260906/smoke_class_T2_scribble | `best.pt`, `last.pt` | 仅 best/state 或诊断权重；最终状态未确认 |
| jiangsuiyang | independent80_seed42_20260906/smoke_domain_A_full | `best.pt`, `last.pt` | 仅 best/state 或诊断权重；最终状态未确认 |
| jiangsuiyang | organ_CL_throughput_20260908/spatial_off/run | `s01_best.pt`, `s01.pt`, `s02_best.pt`, `s02.pt`, `s03_best.pt`, `s03.pt` | 存在阶段权重；待按原协议匹配 |
| jiangsuiyang | organ_T13_half_cl_20260908/balance_checks_20260908_v2/balanced_current_full_replay/run | — | 本次目标目录顶层未检索到 checkpoint |
| jiangsuiyang | organ_T13_half_cl_20260908/balance_checks_20260908_v2/feature_replay_only/run | — | 本次目标目录顶层未检索到 checkpoint |
| jiangsuiyang | organ_T13_half_cl_20260908/balance_checks_20260908_v2/no_replay_losses/run | — | 本次目标目录顶层未检索到 checkpoint |
| jiangsuiyang | organ_T13_half_cl_20260908/balance_checks_20260908_v2/supervision_replay_only/run | — | 本次目标目录顶层未检索到 checkpoint |
| jiangsuiyang | organ_T13_half_cl_20260908/diagnostic_t2_seed43_v3/run | `FIRST_NONFINITE.pt` | 仅 best/state 或诊断权重；最终状态未确认 |
| jiangsuiyang | organ_T13_half_cl_20260908/formal_controls_20260908/alpha0 | `s02_best.pt`, `s02.pt`, `s03_best.pt`, `FIRST_NONFINITE.pt` | 存在阶段权重；待按原协议匹配 |
| jiangsuiyang | organ_T13_half_cl_20260908/formal_controls_20260908/alpha01 | `s02_best.pt`, `s02.pt`, `s03_best.pt`, `FIRST_NONFINITE.pt` | 存在阶段权重；待按原协议匹配 |
| jiangsuiyang | organ_T13_half_cl_20260908/formal_controls_20260908/no_spatial | `s02_best.pt`, `s02.pt`, `s03_best.pt`, `FIRST_NONFINITE.pt` | 存在阶段权重；待按原协议匹配 |
| jiangsuiyang | organ_T13_half_cl_20260908/run | `s01_best.pt` | 仅 best/state 或诊断权重；最终状态未确认 |
| jiangsuiyang | organ_T13_half_cl_20260908/run60 | `s01_best.pt`, `s01.pt` | 存在阶段权重；待按原协议匹配 |
| jiangsuiyang | organ_T13_half_cl_20260908/short_checks_20260908/clean_bn/run | `FIRST_NONFINITE.pt` | 仅 best/state 或诊断权重；最终状态未确认 |
| jiangsuiyang | organ_T13_half_cl_20260908/short_checks_20260908/clean_bn_clip5_126/run | — | 本次目标目录顶层未检索到 checkpoint |
| jiangsuiyang | organ_T13_half_cl_20260908/short_checks_20260908/clip5/run | — | 本次目标目录顶层未检索到 checkpoint |
| jiangsuiyang | organ_T13_half_cl_20260908/short_checks_20260908/clip5_126/run | — | 本次目标目录顶层未检索到 checkpoint |
| jiangsuiyang | organ_T13_half_cl_20260908/short_checks_20260908/clip5_420/run | — | 本次目标目录顶层未检索到 checkpoint |
| jiangsuiyang | organ_T13_half_cl_20260908/short_checks_20260908/clip5_seed44_126/run | — | 本次目标目录顶层未检索到 checkpoint |
| jiangsuiyang | organ_T13_half_cl_20260908/short_checks_20260908/frozen_bn_126/run | `FIRST_NONFINITE.pt` | 仅 best/state 或诊断权重；最终状态未确认 |
| jiangsuiyang | organ_T13_half_cl_20260908/short_checks_20260908/frozen_bn_clip5_126/run | — | 本次目标目录顶层未检索到 checkpoint |
| jiangsuiyang | organ_T13_half_cl_20260908/short_checks_20260908/lr003/run | — | 本次目标目录顶层未检索到 checkpoint |
| jiangsuiyang | organ_T13_half_cl_20260908/short_checks_20260908/lr003_126/run | — | 本次目标目录顶层未检索到 checkpoint |
| jiangsuiyang | organ_T13_half_cl_20260908/short_checks_20260908/lr003_clip5/run | — | 本次目标目录顶层未检索到 checkpoint |
| jiangsuiyang | organ_T13_half_cl_20260908/short_checks_20260908/lr003_clip5_126/run | — | 本次目标目录顶层未检索到 checkpoint |
| jiangsuiyang | organ_T13_half_cl_20260908/small_alpha_checks_20260908/alpha_0.01/run | — | 本次目标目录顶层未检索到 checkpoint |
| jiangsuiyang | organ_T13_half_cl_20260908/small_alpha_checks_20260908/alpha_0.05/run | — | 本次目标目录顶层未检索到 checkpoint |
| jiangsuiyang | organ_T13_half_cl_20260908/small_alpha_checks_20260908/alpha_0.1/run | — | 本次目标目录顶层未检索到 checkpoint |
| jiangsuiyang | organ_T2_coverage_20260908/runs/cl_fg20 | `s01_best.pt` | 仅 best/state 或诊断权重；最终状态未确认 |
| jiangsuiyang | organ_T2_coverage_20260908/runs/cl_fg40 | `s01_best.pt` | 仅 best/state 或诊断权重；最终状态未确认 |
| jiangsuiyang | organ_T2_coverage_20260908/runs/stability_gate | `s01_best.pt`, `s01.pt` | 存在阶段权重；待按原协议匹配 |
| jiangsuiyang | organ_T2_coverage_20260908/runs/sw_lr01_g0 | `s01_best.pt`, `s01.pt`, `s02_best.pt`, `s02.pt`, `s03_best.pt` | 存在阶段权重；待按原协议匹配 |
| jiangsuiyang | organ_T2_coverage_20260908/runs/sw_lr01_g01 | `s01_best.pt`, `s01.pt`, `s02_best.pt`, `s02.pt` | 存在阶段权重；待按原协议匹配 |
| jiangsuiyang | organ_T2_coverage_20260908/runs/sw_lr03_g0 | `s01_best.pt`, `s01.pt`, `s02_best.pt`, `s02.pt`, `s03_best.pt` | 存在阶段权重；待按原协议匹配 |
| jiangsuiyang | organ_T2_coverage_20260908/runs/sw_lr03_g01 | `s01_best.pt`, `s01.pt`, `s02_best.pt`, `s02.pt` | 存在阶段权重；待按原协议匹配 |
| jiangsuiyang | organ_T34_lr006_spatial_pair20_20260910/runs/spatial0 | `s01.pt`, `s02.pt` | 存在阶段权重；待按原协议匹配 |
| jiangsuiyang | organ_T34_lr006_spatial_pair20_20260910/runs/spatial001 | `s01.pt`, `s02.pt` | 存在阶段权重；待按原协议匹配 |
| jiangsuiyang | organ_batch8_probe_20260908/b8_spatial_on/run | — | 本次目标目录顶层未检索到 checkpoint |
| jiangsuiyang | organ_metrics_half_20260909/prefix_T2 | `s01.pt`, `s02.pt` | 存在阶段权重；待按原协议匹配 |
| jiangsuiyang | organ_metrics_half_20260909/runs/ind_T4 | `s01_best.pt` | 仅 best/state 或诊断权重；最终状态未确认 |
| jiangsuiyang | organ_t3_retention_probe_20260909/R0 | `initial_T3_head.pt` | 仅 best/state 或诊断权重；最终状态未确认 |
| jiangsuiyang | organ_t3_retention_probe_20260909/R1 | `initial_T3_head.pt` | 仅 best/state 或诊断权重；最终状态未确认 |
| jiangsuiyang | organ_t3_retention_probe_20260909/R2 | `initial_T3_head.pt` | 仅 best/state 或诊断权重；最终状态未确认 |
| jiangsuiyang | organ_t3_retention_probe_20260909/R3 | `initial_T3_head.pt` | 仅 best/state 或诊断权重；最终状态未确认 |
| jiangsuiyang | organ_t3_retention_probe_20260909/frozen | `initial_T3_head.pt` | 仅 best/state 或诊断权重；最终状态未确认 |
| jiangsuiyang | replay_comparisons_20260911/runs/domain_zs-der | `s01_best.pt`, `FIRST_NONFINITE.pt` | 仅 best/state 或诊断权重；最终状态未确认 |
| jiangsuiyang | replay_comparisons_20260911/runs/domain_zs-er | `s01_best.pt`, `s01.pt`, `s02_best.pt`, `s02.pt`, `s03_best.pt`, `s03.pt`, `s04_best.pt`, `s04.pt`, `s05_best.pt`, `s05.pt`, `s06_best.pt`, `FIRST_NONFINITE.pt` | 存在阶段权重；待按原协议匹配 |
| jiangsuiyang | replay_comparisons_20260911/runs/organ_zs-der | `s01_best.pt`, `FIRST_NONFINITE.pt` | 仅 best/state 或诊断权重；最终状态未确认 |
| jiangsuiyang | tune_independent_A_07261_seed42_20260907_1303/independent | `best.pt` | 仅 best/state 或诊断权重；最终状态未确认 |
| jiangsuiyang | tune_independent_A_07261_seed42_20260907_1303/oracle | `s01_best.pt` | 仅 best/state 或诊断权重；最终状态未确认 |
| jiangsuiyang | tune_independent_A_07261_seed42_20260907_1303/parity/independent | — | 本次目标目录顶层未检索到 checkpoint |
| jiangsuiyang | tune_independent_A_07261_seed42_20260907_1303/parity/oracle | — | 本次目标目录顶层未检索到 checkpoint |
| jiangsuiyang | tune_independent_A_07261_seed42_20260907_1303/parity_deterministic/independent | — | 本次目标目录顶层未检索到 checkpoint |
| jiangsuiyang | tune_independent_A_07261_seed42_20260907_1303/parity_deterministic/oracle | — | 本次目标目录顶层未检索到 checkpoint |
| jiangsuiyang | tune_independent_A_07261_seed42_20260907_1303/parity_pattern_f5_b10/independent | — | 本次目标目录顶层未检索到 checkpoint |
| jiangsuiyang | tune_independent_A_07261_seed42_20260907_1303/parity_pattern_f5_b10/oracle | — | 本次目标目录顶层未检索到 checkpoint |
| jiangsuiyang | tune_independent_A_07261_seed42_20260907_1303/parity_pattern_f5_b10/oracle_repeat | — | 本次目标目录顶层未检索到 checkpoint |
| jiangsuiyang | tune_independent_A_07261_seed42_20260907_1303/parity_repeat/independent | — | 本次目标目录顶层未检索到 checkpoint |
| jiangsuiyang | tune_independent_A_07261_seed42_20260907_1303/parity_repeat/oracle | — | 本次目标目录顶层未检索到 checkpoint |
| jiangsuiyang | tune_independent_A_07261_seed42_20260907_1303/parity_repeat/oracle_repeat | — | 本次目标目录顶层未检索到 checkpoint |
| jiangsuiyang | zs_independent_search_seed42_20260906_inclusive_aborted_20260906_2128/tuning/class_T2_c01 | — | 本次目标目录顶层未检索到 checkpoint |
| jiangsuiyang | zs_independent_search_seed42_20260906_inclusive_aborted_20260906_2128/tuning/class_T2_c02 | — | 本次目标目录顶层未检索到 checkpoint |
| jiangsuiyang | zs_independent_search_seed42_20260906_inclusive_aborted_20260906_2128/tuning/class_T2_c03 | — | 本次目标目录顶层未检索到 checkpoint |
| jiangsuiyang | zs_independent_search_seed42_20260906_inclusive_aborted_20260906_2128/tuning/class_T2_c04 | — | 本次目标目录顶层未检索到 checkpoint |
| jiangsuiyang | zs_independent_search_seed42_20260906_inclusive_aborted_20260906_2128/tuning/domain_C_c01 | `best.pt` | 仅 best/state 或诊断权重；最终状态未确认 |
| jiangsuiyang | zs_independent_search_seed42_20260906_inclusive_aborted_20260906_2128/tuning/domain_C_c02 | `best.pt` | 仅 best/state 或诊断权重；最终状态未确认 |
| jiangsuiyang | zs_independent_search_seed42_20260906_inclusive_aborted_20260906_2128/tuning/domain_C_c03 | `best.pt` | 仅 best/state 或诊断权重；最终状态未确认 |
| jiangsuiyang | zs_independent_search_seed42_20260906_inclusive_aborted_20260906_2128/tuning/domain_C_c04 | `best.pt` | 仅 best/state 或诊断权重；最终状态未确认 |
| jiangsuiyang | zs_independent_tuning_seed42_20260906/smoke_class_T2 | `best.pt`, `last.pt` | 仅 best/state 或诊断权重；最终状态未确认 |
| jiangsuiyang | zs_independent_tuning_seed42_20260906/smoke_domain_B_skiptest | `best.pt`, `last.pt` | 仅 best/state 或诊断权重；最终状态未确认 |

历史检索另外确认：NAS runs/t4m7b 保留 s01–s06，runs/v6c43 保留 s01–s03；旧 q1d7f 的 m7v2q 保留 s01–s06，7 个缺背景的 Joint 短 run 保留 s01.pt。u5k2n 保留前三阶段完成权重及 T4 best，仍不能标为完整训练。旧 q2m8v 的 home/runs 目录为空，实际 Class 成果位于 NAS runs。早期 a2m7x–h9t6z、j3p7n–p6z3t 参数搜索在此次核对的历史 roots 中未定位到对应 run 权重，继续保留 NA；不宣称已经删除或全服务器不存在。工程 smoke、合成数据、仅失败快照不因找到了权重就进入正式性能表。

## 9. 结果使用结论与缺项

1. 当前同预算 Class 正式组已具备完整含背景结果、阶段矩阵与统一独立参考；ZS-DER++-MiB 的 A-Dice=0.767664。ER/guarded DER 已完成，但训练预算不同且旧任务结果弱。
2. Organ ER 已完成且含背景 A-Dice=0.784476。主方法 ZS-DER++ / spatial0 已补评，含背景 A-Dice=0.848059；其预算为 60/60/10/10，ER 为 40/40/40/40，不能直接当作匹配对照。其余 Organ baseline 训练已完成，含背景补评待完成。u5k2n 的 0.768314 来自未完成 checkpoint。
3. 归档 Domain DER++ 为 0.768733，GPM 为 0.657588，但预算分别为 80 与 150 epochs/任务。新 Domain ER/DER 失败，不能用缺失结果代表 0 分或宣称完成。
4. 独立训练、验证筛选、测试选模演示、短诊断与正式持续学习分别展示。所有缺项均保留，不用背景≈1 的假设估算，不复制前景 BWTR/RMA 冒充含背景结果。
5. 后续请求已授权并启动背景补评。原来未学习的任务、原来没有做 test 的验证筛选，以及没有完整 checkpoint 的失败实验，其 NA 不应通过扩大评估范围或猜测填满。补评状态与 checkpoint 索引见文末补充节。

## 10. 来源索引

`local` 为本项目本地归档；两个服务器路径相对其 ScribbleCL 存储根；`published` 为 `DLwbm123/ScribbleCL44` 历史公开归档。来源是产物定位信息，不暗示所有原始文件公开可访问。每个主表数值均取自显式含背景字段，或由这些字段进行均值/比率重算。未发布医学原始数据或模型。

| 编号 | 来源位置 | 相对路径 |
| --- | --- | --- |
| S001 | my-gpu | `class_comparisons_20260910_neutral_restart/jobs/pce_sequential/summary.json` |
| S002 | my-gpu | `class_comparisons_20260910_neutral_restart/jobs/dense_sequential/summary.json` |
| S003 | my-gpu | `class_comparisons_20260910_neutral_restart/jobs/zs_sequential/summary.json` |
| S004 | my-gpu | `class_comparisons_20260910_neutral_restart/jobs/zs_ewc/summary.json` |
| S005 | my-gpu | `class_comparisons_20260910_neutral_restart/jobs/zs_gpm/summary.json` |
| S006 | my-gpu | `class_comparisons_20260910_neutral_restart/jobs/main_s0/summary.json` |
| S007 | my-gpu | `class_replay_60e_20260913/jobs/zs-er/summary.json` |
| S008 | my-gpu | `class_replay_60e_20260913/jobs/zs-der/summary.json` |
| S009 | my-gpu | `class_comparisons_20260910_neutral_restart/jobs/ind_T2_s0/summary.json` |
| S010 | my-gpu | `class_comparisons_20260910_neutral_restart/jobs/ind_T3_s0/summary.json` |
| S011 | jiangsuiyang | `table_metrics_20260910/class_s03.json` |
| S012 | published | `results/class_zs_mib/background_comparison.json` |
| S013 | jiangsuiyang | `replay_comparisons_20260911/runs/organ_zs-er/summary.json` |
| S014 | published | `results/organ_zs_derpp_partial/background_comparison.json` |
| S015 | jiangsuiyang | `background_recovery_20260915/cached/results/r001.json` |
| S016 | jiangsuiyang | `organ_comparisons_20260910/runs/dense-sequential/summary.json` |
| S017 | jiangsuiyang | `organ_comparisons_20260910/runs/pce-sequential/summary.json` |
| S018 | jiangsuiyang | `organ_comparisons_20260910/runs/zs-sequential/summary.json` |
| S019 | jiangsuiyang | `organ_comparisons_20260910/runs/zs-ewc/summary.json` |
| S020 | jiangsuiyang | `organ_comparisons_20260910/runs/zs-gpm/summary.json` |
| S021 | jiangsuiyang | `organ_baseline_tuning_20260911/runs/zs-sequential_formal/summary.json` |
| S022 | jiangsuiyang | `organ_baseline_tuning_20260911/runs/zs-ewc_formal/summary.json` |
| S023 | jiangsuiyang | `organ_baseline_tuning_20260911/runs/zs-gpm_formal/summary.json` |
| S024 | jiangsuiyang | `organ_T3_lr006_spatial_pair10_20260910/runs/spatial0/summary.json` |
| S025 | jiangsuiyang | `organ_T3_lr006_spatial_pair10_20260910/runs/spatial001/summary.json` |
| S026 | jiangsuiyang | `organ_T4_from_T3best_lr006_spatial_pair10_20260910/runs/spatial001/summary.json` |
| S027 | jiangsuiyang | `organ_baseline_tuning_20260911/runs/zs-ewc_c0/summary.json` |
| S028 | jiangsuiyang | `organ_baseline_tuning_20260911/runs/zs-ewc_c1/summary.json` |
| S029 | jiangsuiyang | `organ_baseline_tuning_20260911/runs/zs-ewc_c2/summary.json` |
| S030 | jiangsuiyang | `organ_baseline_tuning_20260911/runs/zs-ewc_c3/summary.json` |
| S031 | jiangsuiyang | `organ_baseline_tuning_20260911/runs/zs-gpm_c0/summary.json` |
| S032 | jiangsuiyang | `organ_baseline_tuning_20260911/runs/zs-gpm_c1/summary.json` |
| S033 | jiangsuiyang | `organ_baseline_tuning_20260911/runs/zs-gpm_c2/summary.json` |
| S034 | jiangsuiyang | `organ_baseline_tuning_20260911/runs/zs-gpm_c3/summary.json` |
| S035 | jiangsuiyang | `organ_baseline_tuning_20260911/runs/zs-sequential_c0/summary.json` |
| S036 | jiangsuiyang | `organ_baseline_tuning_20260911/runs/zs-sequential_c1/summary.json` |
| S037 | jiangsuiyang | `organ_baseline_tuning_20260911/runs/zs-sequential_c2/summary.json` |
| S038 | jiangsuiyang | `organ_baseline_tuning_20260911/runs/zs-sequential_c3/summary.json` |
| S039 | jiangsuiyang | `organ_metrics_T134_half_20260909/runs/ind_T4/summary.json` |
| S040 | jiangsuiyang | `organ_metrics_half_20260909/runs/ind_T3/summary.json` |
| S041 | jiangsuiyang | `organ_metrics_half_20260909/runs/organ_no_replay/summary.json` |
| S042 | jiangsuiyang | `organ_metrics_half_20260909/runs/organ_retention/summary.json` |
| S043 | jiangsuiyang | `replay_comparisons_20260911/progress.json` |
| S044 | local | `independent_a_work/results/domain_zs_derpp/background_comparison.json` |
| S045 | local | `independent_a_work/results/domain_zs_gpm/background_comparison.json` |
| S046 | local | `independent_a_work/results/joint_short_20260902/background_comparison.json` |
| S047 | my-gpu | `class_replay_improve_20260915_r2/progress.json` |
| S048 | my-gpu | `background_recovery_20260915/extra/results/r003.json` |
| S049 | my-gpu | `background_recovery_20260915/extra/results/r004.json` |
| S050 | my-gpu | `background_recovery_20260915/extra/results/r001.json` |
| S051 | my-gpu | `background_recovery_20260915/extra/results/r002.json` |
| S052 | local | `independent_a_work/results/independent_training_80e_seed42_20260906.csv` |
| S053 | local | `independent_a_work/results/zs_independent_80e_seed42_20260907.csv` |
| S054 | local | `independent_a_work/results/domain_zs_independent_matched_80e_seed42_20260907.csv` |
| S055 | local | `independent_a_work/results/zs_independent_tuning_20e_seed42_20260907.csv` |
| S056 | jiangsuiyang | `independent_A_spatial_sweep20_formal80_20260907/formal80/summary.json` |
| S057 | jiangsuiyang | `independent_A_spatial_sweep20_formal80_20260907/sweep_s00/summary.json` |
| S058 | jiangsuiyang | `independent_A_spatial_sweep20_formal80_20260907/sweep_s01/summary.json` |
| S059 | jiangsuiyang | `independent_A_spatial_sweep20_formal80_20260907/sweep_s02/summary.json` |
| S060 | jiangsuiyang | `independent_A_spatial_sweep20_formal80_20260907/sweep_s03/summary.json` |
| S061 | jiangsuiyang | `independent_A_spatial_sweep20_formal80_20260907/sweep_s04/summary.json` |
| S062 | jiangsuiyang | `independent_A_spatial_sweep20_formal80_20260907/sweep_s05/summary.json` |
| S063 | jiangsuiyang | `independent_A_warmup_sweep80_20260907/warmup10/summary.json` |
| S064 | jiangsuiyang | `independent_A_warmup_sweep80_20260907/warmup20/summary.json` |
| S065 | jiangsuiyang | `independent_A_warmup_sweep80_20260907/warmup40/summary.json` |
| S066 | jiangsuiyang | `independent_A_warmup_sweep80_20260907/warmup60/summary.json` |
| S067 | jiangsuiyang | `independent_BF_formal80_warmup10_20260907/B/summary.json` |
| S068 | jiangsuiyang | `independent_BF_formal80_warmup10_20260907/C/summary.json` |
| S069 | jiangsuiyang | `independent_BF_formal80_warmup10_20260907/D/summary.json` |
| S070 | jiangsuiyang | `independent_BF_formal80_warmup10_20260907/E/summary.json` |
| S071 | jiangsuiyang | `independent_BF_formal80_warmup10_20260907/F/summary.json` |
| S072 | jiangsuiyang | `independent_B_shared_smoke_20260907/summary.json` |
| S073 | jiangsuiyang | `independent_demo_smoke_20260907/summary.json` |
| S074 | jiangsuiyang | `organ_T2_original_scribble_20260908/runs/organ_fg20/summary.json` |
| S075 | jiangsuiyang | `organ_T2_reference_recovery_20260908/runs/domain_control/summary.json` |
| S076 | jiangsuiyang | `organ_T2_reference_recovery_20260908/runs/organ_reference/summary.json` |
| S077 | jiangsuiyang | `organ_T2_spatial30_deterministic_20260908/runs/domain_control/summary.json` |
| S078 | jiangsuiyang | `organ_T2_spatial30_deterministic_20260908/runs/organ_reference/summary.json` |
| S079 | jiangsuiyang | `organ_T2_spatial_start30_20260908/runs/domain_control/summary.json` |
| S080 | jiangsuiyang | `organ_T2_spatial_start30_20260908/runs/organ_reference/summary.json` |
| S081 | jiangsuiyang | `spatial_sweep_smoke_20260907/summary.json` |
| S082 | jiangsuiyang | `tune_independent_A_07261_seed42_20260907_1303/independent_pattern_f5_b10/summary.json` |
| S083 | jiangsuiyang | `tune_independent_A_07261_seed42_20260907_1303/smoke/summary.json` |
| S084 | local | `independent_a_work/results/independent_domains_demo_test_selection_20260907/results.csv` |
| S085 | local | `independent_a_work/results/independent_domains_sweep4_lr_global_20260908/all_runs.csv` |
| S086 | my-gpu | `class_comparisons_20260910/dense_smoke/summary.json` |
| S087 | my-gpu | `class_comparisons_20260910/gpm_smoke/summary.json` |
| S088 | my-gpu | `background_recovery_20260915/results/r015.json` |
| S089 | my-gpu | `background_recovery_20260915/results/r005.json` |
| S090 | my-gpu | `background_recovery_20260915/results/r006.json` |
| S091 | my-gpu | `background_recovery_20260915/results/r007.json` |
| S092 | my-gpu | `background_recovery_20260915/results/r008.json` |
| S093 | my-gpu | `background_recovery_20260915/results/r009.json` |
| S094 | my-gpu | `background_recovery_20260915/results/r010.json` |
| S095 | my-gpu | `class_independent_spatial_sweep_20260908/smoke_cl_spatial_original_gco/summary.json` |
| S096 | my-gpu | `class_independent_spatial_sweep_20260908/smoke_ind_T3_gco/summary.json` |
| S097 | my-gpu | `class_replay_improve_20260915/jobs/zs-der-smoke/summary.json` |
| S098 | my-gpu | `class_replay_improve_20260915_r2/jobs/zs-der-smoke/summary.json` |
| S099 | my-gpu | `class_replay_improve_20260915_r2/jobs/zs-er-smoke/summary.json` |
| S100 | my-gpu | `background_recovery_20260915/results/r011.json` |
| S101 | my-gpu | `background_recovery_20260915/results/r012.json` |
| S102 | my-gpu | `background_recovery_20260915/results/r013.json` |
| S103 | my-gpu | `background_recovery_20260915/results/r014.json` |
| S104 | my-gpu | `domain_joint_runs_20260901/runs/j0b2l15_r1/summary.json` |
| S105 | my-gpu | `domain_joint_runs_20260901/runs/j1b4l30_r1/summary.json` |
| S106 | my-gpu | `background_recovery_20260915/results/r016.json` |
| S107 | my-gpu | `background_recovery_20260915/results/r017.json` |
| S108 | my-gpu | `domain_runs_20260901/smoke/joint_smoke_b8/summary.json` |
| S109 | my-gpu | `scribblecl_domain_organ_20260809/runs/domain/domain_pce_ft_seed42_20260809T082951Z/summary.json` |
| S110 | my-gpu | `scribblecl_domain_organ_20260809/runs/organ/organ_pce_ft_seed42_20260809T083415Z/summary.json` |
| S111 | jiangsuiyang | `class_independent_spatial_sweep_20260908/smoke_cl_spatial_chunked/summary.json` |
| S112 | jiangsuiyang | `class_independent_spatial_sweep_20260908/smoke_ind_T3/summary.json` |
| S113 | jiangsuiyang | `organ_CL_throughput_20260908/memory_spatial_off/run/summary.json` |
| S114 | jiangsuiyang | `organ_CL_throughput_20260908/memory_spatial_on/run/summary.json` |
| S115 | jiangsuiyang | `organ_T13_half_cl_20260908/balance_checks_20260908_v2/summary.json` |
| S116 | jiangsuiyang | `organ_T13_half_cl_20260908/diagnostic_t2_seed43_v3/summary.json` |
| S117 | jiangsuiyang | `organ_T13_half_cl_20260908/formal_small_alpha_20260908/summary.json` |
| S118 | jiangsuiyang | `organ_T13_half_cl_20260908/short_checks_20260908/summary.json` |
| S119 | jiangsuiyang | `organ_T13_half_cl_20260908/subset/summary.json` |
| S120 | jiangsuiyang | `organ_T13_half_cl_20260908/task_strategy_check/synthetic/run/summary.json` |
| S121 | jiangsuiyang | `organ_T13_half_cl_20260908/task_strategy_check/synthetic_small_alpha/resume_t2/summary.json` |
| S122 | jiangsuiyang | `organ_T13_half_cl_20260908/task_strategy_check/synthetic_small_alpha/run/summary.json` |
| S123 | jiangsuiyang | `organ_T2_coverage_20260908/public_release/summary.json` |
| S124 | jiangsuiyang | `organ_T2_coverage_20260908/runs/e1_sw_lr01_g0/summary.json` |
| S125 | jiangsuiyang | `organ_T2_coverage_20260908/runs/e1_sw_lr01_g01/summary.json` |
| S126 | jiangsuiyang | `organ_T2_coverage_20260908/runs/e1_sw_lr03_g0/summary.json` |
| S127 | jiangsuiyang | `organ_T2_coverage_20260908/runs/e1_sw_lr03_g01/summary.json` |
| S128 | jiangsuiyang | `organ_T2_coverage_20260908/runs/ind_fg20/summary.json` |
| S129 | jiangsuiyang | `organ_T2_coverage_20260908/runs/ind_fg40/summary.json` |
| S130 | jiangsuiyang | `organ_T2_coverage_20260908/runs/stability_gate_mb4/summary.json` |
| S131 | jiangsuiyang | `organ_domain_swap_20260909/bidmc/t2/summary.json` |
| S132 | jiangsuiyang | `organ_domain_swap_20260909/bidmc/t3/summary.json` |
| S133 | jiangsuiyang | `organ_domain_swap_20260909/synthetic_check/resume_t2/summary.json` |
| S134 | jiangsuiyang | `organ_domain_swap_20260909/synthetic_check/run/summary.json` |
| S135 | jiangsuiyang | `organ_domain_swap_20260909/ucl/t2/summary.json` |
| S136 | jiangsuiyang | `organ_domain_swap_20260909/ucl/t3/summary.json` |
| S137 | jiangsuiyang | `organ_t3_retention_probe_20260909/delivery/summary.json` |
| S138 | jiangsuiyang | `tune_independent_A_07261_seed42_20260907_1303/oracle_pattern_f5_b10/summary.json` |
| S139 | local | `organ_annotation_work/results/zs_derpp_ab_sweep_20260902/metrics.csv` |
| S140 | local | `organ_annotation_work/results/organ_zs_derpp_ab_sweep_20260902/metrics.csv` |
| S141 | local | `organ_annotation_work/results/joint_short_20260902/metrics.csv` |
| S142 | jiangsuiyang | `background_recovery_20260915/legacy/results/l001.json` |
| S143 | jiangsuiyang | `background_recovery_20260915/legacy/results/l002.json` |
| S144 | jiangsuiyang | `background_recovery_20260915/legacy/results/l003.json` |
| S145 | jiangsuiyang | `background_recovery_20260915/legacy/results/l004.json` |
| S146 | jiangsuiyang | `background_recovery_20260915/legacy/results/l005.json` |
| S147 | jiangsuiyang | `background_recovery_20260915/legacy/results/l006.json` |
| S148 | jiangsuiyang | `background_recovery_20260915/legacy/results/l007.json` |
| S149 | my-gpu | `static_reference_exports_20260807/static_reference_results.csv` |
| S150 | my-gpu | `coverage_runs/B1/pce_seed42_stage1/stage_metrics.csv` |
| S151 | my-gpu | `coverage_runs/B1/zs_seed42_stage1/stage_metrics.csv` |
| S152 | my-gpu | `coverage_runs/B2/pce_seed42_stage1/stage_metrics.csv` |
| S153 | my-gpu | `coverage_runs/B2/zs_seed42_stage1/stage_metrics.csv` |
| S154 | my-gpu | `coverage_runs/B3/pce_seed42_stage1/stage_metrics.csv` |
| S155 | my-gpu | `coverage_runs/B3/zs_seed42_stage1/stage_metrics.csv` |
| S156 | my-gpu | `coverage_runs/Dense/dense_seed42_stage1/stage_metrics.csv` |
| S157 | my-gpu | `overfit/pce_mib_seed42_stage2/stage_metrics.csv` |
| S158 | my-gpu | `overfit/pce_seed42_stage1/stage_metrics.csv` |
| S159 | my-gpu | `overfit/zs_mib_seed42_stage2/stage_metrics.csv` |
| S160 | my-gpu | `overfit/zs_seed42_stage1/stage_metrics.csv` |
| S161 | my-gpu | `overfit_fixed/pce_mib_seed42_stage2/stage_metrics.csv` |
| S162 | my-gpu | `class_comparisons_20260910/jobs/pce_sequential/manifest.json` |
| S163 | my-gpu | `class_comparisons_20260910/jobs/zs_sequential/manifest.json` |
| S164 | my-gpu | `class_er_gpu3_20260913/jobs/zs-er/manifest.json` |
| S165 | my-gpu | `class_independent_spatial_sweep_20260908/jobs/cl_spatial_0.001/manifest.json` |
| S166 | my-gpu | `class_independent_spatial_sweep_20260908/jobs/ind_T1_formal/manifest.json` |
| S167 | my-gpu | `class_independent_spatial_sweep_20260908/smoke_cl_spatial_original/manifest.json` |
| S168 | my-gpu | `class_independent_spatial_sweep_20260908/smoke_ind_T3/manifest.json` |
| S169 | my-gpu | `class_replay_comparisons_20260912/jobs/zs-der/manifest.json` |
| S170 | my-gpu | `class_replay_comparisons_20260912/jobs/zs-er/manifest.json` |
| S171 | my-gpu | `class_replay_improve_20260915/jobs/zs-der-control5/manifest.json` |
| S172 | my-gpu | `class_replay_improve_20260915/jobs/zs-er-smoke/manifest.json` |
| S173 | my-gpu | `class_replay_resume_20260913/jobs/zs-der/manifest.json` |
| S174 | my-gpu | `class_replay_resume_20260913/jobs/zs-er/manifest.json` |
| S175 | my-gpu | `core_runs/class_q8v2n6_seed42_150e_20260811T1345Z/zs_mib_raw_global1_diagnostic_stopped_iter1000_seed42/manifest.json` |
| S176 | my-gpu | `core_runs/class_q8v2n6_seed42_150e_20260811T1345Z/zs_sequential_raw_global1_diagnostic_stopped_iter1200_seed42/manifest.json` |
| S177 | my-gpu | `domain_joint_c3fix_20260902/runs/c3adam_e5_s42/manifest.json` |
| S178 | my-gpu | `domain_joint_c3fix_20260902/runs/c3adamgd_e5_s42/manifest.json` |
| S179 | my-gpu | `domain_joint_c3fix_20260902/runs/c3sgd_e3_s42/manifest.json` |
| S180 | my-gpu | `domain_joint_validation_20260902/runs/balpce_b4e20_s42/manifest.json` |
| S181 | my-gpu | `domain_joint_validation_20260902/runs/balzs_b4e20_s42/manifest.json` |
| S182 | my-gpu | `organ_runs_20260901/runs/o1seq_r1/manifest.json` |
| S183 | my-gpu | `organ_runs_20260901/runs/o2ewc_r1/manifest.json` |
| S184 | my-gpu | `organ_runs_20260901/runs/u2k7m/manifest.json` |
| S185 | my-gpu | `organ_runs_20260901/runs/u2k7m_r1/manifest.json` |
| S186 | my-gpu | `organ_runs_20260901/runs/u2k7m_r2/manifest.json` |
| S187 | my-gpu | `organ_runs_20260901/runs/v3p8n/manifest.json` |
| S188 | my-gpu | `organ_runs_20260901/runs/v3p8n_r1/manifest.json` |
| S189 | my-gpu | `organ_runs_20260901/runs/v3p8n_r2/manifest.json` |
| S190 | jiangsuiyang | `class_independent_spatial_sweep_20260908/jobs/ind_T1_lr0.01/manifest.json` |
| S191 | jiangsuiyang | `class_independent_spatial_sweep_20260908/jobs/ind_T2_lr0.01/manifest.json` |
| S192 | jiangsuiyang | `class_independent_spatial_sweep_20260908/jobs/ind_T3_lr0.01/manifest.json` |
| S193 | jiangsuiyang | `class_independent_spatial_sweep_20260908/smoke_cl_spatial/manifest.json` |
| S194 | jiangsuiyang | `independent80_seed42_20260906/smoke_class_T2_scribble/manifest.json` |
| S195 | jiangsuiyang | `independent80_seed42_20260906/smoke_domain_A_full/manifest.json` |
| S196 | jiangsuiyang | `organ_CL_throughput_20260908/spatial_off/run/manifest.json` |
| S197 | jiangsuiyang | `organ_T13_half_cl_20260908/balance_checks_20260908_v2/balanced_current_full_replay/run/manifest.json` |
| S198 | jiangsuiyang | `organ_T13_half_cl_20260908/balance_checks_20260908_v2/feature_replay_only/run/manifest.json` |
| S199 | jiangsuiyang | `organ_T13_half_cl_20260908/balance_checks_20260908_v2/no_replay_losses/run/manifest.json` |
| S200 | jiangsuiyang | `organ_T13_half_cl_20260908/balance_checks_20260908_v2/supervision_replay_only/run/manifest.json` |
| S201 | jiangsuiyang | `organ_T13_half_cl_20260908/diagnostic_t2_seed43_v3/run/manifest.json` |
| S202 | jiangsuiyang | `organ_T13_half_cl_20260908/formal_controls_20260908/alpha0/manifest.json` |
| S203 | jiangsuiyang | `organ_T13_half_cl_20260908/formal_controls_20260908/alpha01/manifest.json` |
| S204 | jiangsuiyang | `organ_T13_half_cl_20260908/formal_controls_20260908/no_spatial/manifest.json` |
| S205 | jiangsuiyang | `organ_T13_half_cl_20260908/run/manifest.json` |
| S206 | jiangsuiyang | `organ_T13_half_cl_20260908/run60/manifest.json` |
| S207 | jiangsuiyang | `organ_T13_half_cl_20260908/short_checks_20260908/clean_bn/run/manifest.json` |
| S208 | jiangsuiyang | `organ_T13_half_cl_20260908/short_checks_20260908/clean_bn_clip5_126/run/manifest.json` |
| S209 | jiangsuiyang | `organ_T13_half_cl_20260908/short_checks_20260908/clip5/run/manifest.json` |
| S210 | jiangsuiyang | `organ_T13_half_cl_20260908/short_checks_20260908/clip5_126/run/manifest.json` |
| S211 | jiangsuiyang | `organ_T13_half_cl_20260908/short_checks_20260908/clip5_420/run/manifest.json` |
| S212 | jiangsuiyang | `organ_T13_half_cl_20260908/short_checks_20260908/clip5_seed44_126/run/manifest.json` |
| S213 | jiangsuiyang | `organ_T13_half_cl_20260908/short_checks_20260908/frozen_bn_126/run/manifest.json` |
| S214 | jiangsuiyang | `organ_T13_half_cl_20260908/short_checks_20260908/frozen_bn_clip5_126/run/manifest.json` |
| S215 | jiangsuiyang | `organ_T13_half_cl_20260908/short_checks_20260908/lr003/run/manifest.json` |
| S216 | jiangsuiyang | `organ_T13_half_cl_20260908/short_checks_20260908/lr003_126/run/manifest.json` |
| S217 | jiangsuiyang | `organ_T13_half_cl_20260908/short_checks_20260908/lr003_clip5/run/manifest.json` |
| S218 | jiangsuiyang | `organ_T13_half_cl_20260908/short_checks_20260908/lr003_clip5_126/run/manifest.json` |
| S219 | jiangsuiyang | `organ_T13_half_cl_20260908/small_alpha_checks_20260908/alpha_0.01/run/manifest.json` |
| S220 | jiangsuiyang | `organ_T13_half_cl_20260908/small_alpha_checks_20260908/alpha_0.05/run/manifest.json` |
| S221 | jiangsuiyang | `organ_T13_half_cl_20260908/small_alpha_checks_20260908/alpha_0.1/run/manifest.json` |
| S222 | jiangsuiyang | `organ_T2_coverage_20260908/runs/cl_fg20/manifest.json` |
| S223 | jiangsuiyang | `organ_T2_coverage_20260908/runs/cl_fg40/manifest.json` |
| S224 | jiangsuiyang | `organ_T2_coverage_20260908/runs/stability_gate/manifest.json` |
| S225 | jiangsuiyang | `organ_T2_coverage_20260908/runs/sw_lr01_g0/manifest.json` |
| S226 | jiangsuiyang | `organ_T2_coverage_20260908/runs/sw_lr01_g01/manifest.json` |
| S227 | jiangsuiyang | `organ_T2_coverage_20260908/runs/sw_lr03_g0/manifest.json` |
| S228 | jiangsuiyang | `organ_T2_coverage_20260908/runs/sw_lr03_g01/manifest.json` |
| S229 | jiangsuiyang | `organ_T34_lr006_spatial_pair20_20260910/runs/spatial0/manifest.json` |
| S230 | jiangsuiyang | `organ_T34_lr006_spatial_pair20_20260910/runs/spatial001/manifest.json` |
| S231 | jiangsuiyang | `organ_batch8_probe_20260908/b8_spatial_on/run/manifest.json` |
| S232 | jiangsuiyang | `organ_metrics_half_20260909/prefix_T2/manifest.json` |
| S233 | jiangsuiyang | `organ_metrics_half_20260909/runs/ind_T4/manifest.json` |
| S234 | jiangsuiyang | `organ_t3_retention_probe_20260909/R0/manifest.json` |
| S235 | jiangsuiyang | `organ_t3_retention_probe_20260909/R1/manifest.json` |
| S236 | jiangsuiyang | `organ_t3_retention_probe_20260909/R2/manifest.json` |
| S237 | jiangsuiyang | `organ_t3_retention_probe_20260909/R3/manifest.json` |
| S238 | jiangsuiyang | `organ_t3_retention_probe_20260909/frozen/manifest.json` |
| S239 | jiangsuiyang | `replay_comparisons_20260911/runs/domain_zs-der/manifest.json` |
| S240 | jiangsuiyang | `replay_comparisons_20260911/runs/domain_zs-er/manifest.json` |
| S241 | jiangsuiyang | `replay_comparisons_20260911/runs/organ_zs-der/manifest.json` |
| S242 | jiangsuiyang | `tune_independent_A_07261_seed42_20260907_1303/independent/manifest.json` |
| S243 | jiangsuiyang | `tune_independent_A_07261_seed42_20260907_1303/oracle/manifest.json` |
| S244 | jiangsuiyang | `tune_independent_A_07261_seed42_20260907_1303/parity/independent/manifest.json` |
| S245 | jiangsuiyang | `tune_independent_A_07261_seed42_20260907_1303/parity/oracle/manifest.json` |
| S246 | jiangsuiyang | `tune_independent_A_07261_seed42_20260907_1303/parity_deterministic/independent/manifest.json` |
| S247 | jiangsuiyang | `tune_independent_A_07261_seed42_20260907_1303/parity_deterministic/oracle/manifest.json` |
| S248 | jiangsuiyang | `tune_independent_A_07261_seed42_20260907_1303/parity_pattern_f5_b10/independent/manifest.json` |
| S249 | jiangsuiyang | `tune_independent_A_07261_seed42_20260907_1303/parity_pattern_f5_b10/oracle/manifest.json` |
| S250 | jiangsuiyang | `tune_independent_A_07261_seed42_20260907_1303/parity_pattern_f5_b10/oracle_repeat/manifest.json` |
| S251 | jiangsuiyang | `tune_independent_A_07261_seed42_20260907_1303/parity_repeat/independent/manifest.json` |
| S252 | jiangsuiyang | `tune_independent_A_07261_seed42_20260907_1303/parity_repeat/oracle/manifest.json` |
| S253 | jiangsuiyang | `tune_independent_A_07261_seed42_20260907_1303/parity_repeat/oracle_repeat/manifest.json` |
| S254 | jiangsuiyang | `zs_independent_search_seed42_20260906_inclusive_aborted_20260906_2128/tuning/class_T2_c01/manifest.json` |
| S255 | jiangsuiyang | `zs_independent_search_seed42_20260906_inclusive_aborted_20260906_2128/tuning/class_T2_c02/manifest.json` |
| S256 | jiangsuiyang | `zs_independent_search_seed42_20260906_inclusive_aborted_20260906_2128/tuning/class_T2_c03/manifest.json` |
| S257 | jiangsuiyang | `zs_independent_search_seed42_20260906_inclusive_aborted_20260906_2128/tuning/class_T2_c04/manifest.json` |
| S258 | jiangsuiyang | `zs_independent_search_seed42_20260906_inclusive_aborted_20260906_2128/tuning/domain_C_c01/manifest.json` |
| S259 | jiangsuiyang | `zs_independent_search_seed42_20260906_inclusive_aborted_20260906_2128/tuning/domain_C_c02/manifest.json` |
| S260 | jiangsuiyang | `zs_independent_search_seed42_20260906_inclusive_aborted_20260906_2128/tuning/domain_C_c03/manifest.json` |
| S261 | jiangsuiyang | `zs_independent_search_seed42_20260906_inclusive_aborted_20260906_2128/tuning/domain_C_c04/manifest.json` |
| S262 | jiangsuiyang | `zs_independent_tuning_seed42_20260906/smoke_class_T2/manifest.json` |
| S263 | jiangsuiyang | `zs_independent_tuning_seed42_20260906/smoke_domain_B_skiptest/manifest.json` |
| S264 | jiangsuiyang | `background_recovery_20260915/results/r001.json` |
| S265 | jiangsuiyang | `background_recovery_20260915/results/r002.json` |
| S266 | jiangsuiyang | `background_recovery_20260915/results/r003.json` |
| S267 | jiangsuiyang | `background_recovery_20260915/results/r004.json` |
| S268 | jiangsuiyang | `background_recovery_20260915/results/r005.json` |
| S269 | jiangsuiyang | `background_recovery_20260915/results/r006.json` |
| S270 | jiangsuiyang | `background_recovery_20260915/results/r007.json` |
| S271 | jiangsuiyang | `background_recovery_20260915/results/r008.json` |
| S272 | jiangsuiyang | `background_recovery_20260915/results/r009.json` |
| S273 | jiangsuiyang | `background_recovery_20260915/results/r010.json` |
| S274 | jiangsuiyang | `background_recovery_20260915/results/r011.json` |
| S275 | jiangsuiyang | `background_recovery_20260915/results/r012.json` |
| S276 | jiangsuiyang | `background_recovery_20260915/results/r013.json` |
| S277 | jiangsuiyang | `background_recovery_20260915/results/r014.json` |
| S278 | jiangsuiyang | `background_recovery_20260915/results/r015.json` |
| S279 | jiangsuiyang | `background_recovery_20260915/results/r016.json` |
| S280 | jiangsuiyang | `background_recovery_20260915/results/r017.json` |
| S281 | jiangsuiyang | `background_recovery_20260915/results/r018.json` |
| S282 | jiangsuiyang | `background_recovery_20260915/results/r019.json` |
| S283 | jiangsuiyang | `background_recovery_20260915/results/r020.json` |
| S284 | jiangsuiyang | `background_recovery_20260915/results/r021.json` |
| S285 | jiangsuiyang | `background_recovery_20260915/results/r022.json` |
| S286 | jiangsuiyang | `background_recovery_20260915/results/r023.json` |
| S287 | jiangsuiyang | `background_recovery_20260915/results/r024.json` |
| S288 | jiangsuiyang | `background_recovery_20260915/results/r025.json` |
| S289 | jiangsuiyang | `background_recovery_20260915/results/r026.json` |
| S290 | jiangsuiyang | `background_recovery_20260915/results/r027.json` |
| S291 | jiangsuiyang | `background_recovery_20260915/results/r028.json` |
| S292 | jiangsuiyang | `background_recovery_20260915/legacy/results/l008.json` |
| S293 | jiangsuiyang | `background_recovery_20260915/legacy/results/l009.json` |
| S294 | jiangsuiyang | `background_recovery_20260915/legacy/results/l010.json` |
| S295 | jiangsuiyang | `background_recovery_20260915/legacy/results/l011.json` |
| S296 | jiangsuiyang | `background_recovery_20260915/legacy/results/l012.json` |
| S297 | jiangsuiyang | `background_recovery_20260915/legacy/results/l013.json` |
| S298 | jiangsuiyang | `background_recovery_20260915/legacy/results/l014.json` |
| S299 | jiangsuiyang | `background_recovery_20260915/legacy/results/l015.json` |
| S300 | jiangsuiyang | `background_recovery_20260915/legacy/results/l016.json` |
| S301 | jiangsuiyang | `background_recovery_20260915/legacy/results/l017.json` |
