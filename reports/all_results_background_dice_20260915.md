# 现有实验结果汇总（Dice 包含背景）

汇总日期：2026-09-15（北京时间）。来源包括本地已归档结果、`my-gpu` 与 `jiangsuiyang` 当前可访问实验目录，以及现有公开仓库中的历史含背景复评结果。主表优先采用本次读取的原始汇总产物；后台改进实验状态另见第 5 节。

本文件汇总已有结果，不新启动训练或模型复评。正文给出可读主表，附录保留筛选、诊断、缺项及未形成完整结果的记录。复制的参考目录、发布镜像和每个 epoch 的重复曲线不计作新实验。这里的“已有”是本次可访问产物范围，不代表不可访问、已删除或未归档的实验。

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

### 3.1 已有含背景测试结果

ER 的 4 阶段已完成，采用 T1/T3/T4 训练集减半、T2 保持完整的协议，40 epochs/任务、buffer=64。历史 u5k2n 只是 T4 epoch 21 时保存的 partial checkpoint，不能视为完成的正式对照。

| 方法 / run | 状态 / 预算 | T1 | T2 | T3 | T4 | A-Dice | 来源 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ZS-ER | 完成；40/40/40/40 | 0.715004 | 0.642424 | 0.879449 | 0.901029 | 0.784476 | S013 |
| ZS-DER++ / u5k2n | 未完成；T4 epoch 21 | 0.763569 | 0.497127 | 0.920583 | 0.891977 | 0.768314 | S014 |

| ER 阶段 | T1 | T2 | T3 | T4 |
| --- | --- | --- | --- | --- |
| 1 | 0.809017 | NA | NA | NA |
| 2 | 0.754812 | 0.794554 | NA | NA |
| 3 | 0.695438 | 0.702402 | 0.917629 | NA |
| 4 | 0.715004 | 0.642424 | 0.879449 | 0.901029 |

该 ER 结果的含背景 BWTR=-0.116427。历史 partial 与当前 ER 的训练预算、配置和数据子集不同，不构成匹配对照。

### 3.2 已完成但缺少背景分项的 Organ 批次

以下批次有已保存的前景评估结果，但不足以重算病例级含背景 Dice。A-Dice、WCD 和依赖这些分数的跨任务指标均不填入前景旧值。数据子集、任务策略、学习率与预算变化均需保留。

| 批次 / run | 用途 | 完成阶段 | 配置预算 | 含背景 A-Dice | 来源 |
| --- | --- | --- | --- | --- | --- |
| organ_T3_lr006_spatial_pair10_20260910/runs/spatial0 | 已保存阶段测试 | 3 | 60/60/10 | NA | S015 |
| organ_T3_lr006_spatial_pair10_20260910/runs/spatial001 | 已保存阶段测试 | 3 | 60/60/10 | NA | S016 |
| organ_T4_from_T3best_lr006_spatial_pair10_20260910/runs/spatial0 | 已保存阶段测试 | 4 | 60/60/10/10 | NA | S017 |
| organ_T4_from_T3best_lr006_spatial_pair10_20260910/runs/spatial001 | 已保存阶段测试 | 4 | 60/60/10/10 | NA | S018 |
| organ_baseline_tuning_20260911/runs/zs-ewc_c0 | 筛选验证 | 4 | 10 | NA | S019 |
| organ_baseline_tuning_20260911/runs/zs-ewc_c1 | 筛选验证 | 4 | 10 | NA | S020 |
| organ_baseline_tuning_20260911/runs/zs-ewc_c2 | 筛选验证 | 4 | 10 | NA | S021 |
| organ_baseline_tuning_20260911/runs/zs-ewc_c3 | 筛选验证 | 4 | 10 | NA | S022 |
| organ_baseline_tuning_20260911/runs/zs-ewc_formal | 已保存阶段测试 | 4 | 40 | NA | S023 |
| organ_baseline_tuning_20260911/runs/zs-gpm_c0 | 筛选验证 | 4 | 10 | NA | S024 |
| organ_baseline_tuning_20260911/runs/zs-gpm_c1 | 筛选验证 | 4 | 10 | NA | S025 |
| organ_baseline_tuning_20260911/runs/zs-gpm_c2 | 筛选验证 | 4 | 10 | NA | S026 |
| organ_baseline_tuning_20260911/runs/zs-gpm_c3 | 筛选验证 | 4 | 10 | NA | S027 |
| organ_baseline_tuning_20260911/runs/zs-gpm_formal | 已保存阶段测试 | 4 | 40 | NA | S028 |
| organ_baseline_tuning_20260911/runs/zs-sequential_c0 | 筛选验证 | 4 | 10 | NA | S029 |
| organ_baseline_tuning_20260911/runs/zs-sequential_c1 | 筛选验证 | 4 | 10 | NA | S030 |
| organ_baseline_tuning_20260911/runs/zs-sequential_c2 | 筛选验证 | 4 | 10 | NA | S031 |
| organ_baseline_tuning_20260911/runs/zs-sequential_c3 | 筛选验证 | 4 | 10 | NA | S032 |
| organ_baseline_tuning_20260911/runs/zs-sequential_formal | 已保存阶段测试 | 4 | 40 | NA | S033 |
| organ_comparisons_20260910/runs/dense-sequential | 已保存阶段测试 | 4 | 60/60/10/10 | NA | S034 |
| organ_comparisons_20260910/runs/pce-sequential | 已保存阶段测试 | 4 | 60/60/10/10 | NA | S035 |
| organ_comparisons_20260910/runs/zs-ewc | 已保存阶段测试 | 4 | 60/60/10/10 | NA | S036 |
| organ_comparisons_20260910/runs/zs-gpm | 已保存阶段测试 | 4 | 60/60/10/10 | NA | S037 |
| organ_comparisons_20260910/runs/zs-sequential | 已保存阶段测试 | 4 | 60/60/10/10 | NA | S038 |
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

这里可比较保存 checkpoint 的含背景最终分数，但没有所有阶段的含背景矩阵，故不复用归档 CSV 中按前景计算的 BWTR/E-FWT。9 月 11 日新启动的 Domain ER、DER 均失败，见 3.3；不要用它们覆盖这两项历史完成结果。

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
| zs-er-control5 | 验证集 | 0.566031 | NA | NA | S048 |
| zs-er-improved5 | 验证集 | 0.677205 | NA | NA | S049 |
| zs-der-control5 | 验证集 | 0.514116 | NA | NA | S050 |
| zs-der-improved5 | 验证集 | 0.482276 | NA | NA | S051 |

上述有值的 T2 来自各自已保存 best_validation；旧任务的 seen_validation 只保存前景标量，无法据此算含背景双任务均值。ER 改进筛选通过、DER 改进筛选未通过原门槛，正式提升幅度仍需完成同预算测试后判断。首轮的 int16 gather 索引故障、被修复替代的筛选及 smoke 检查仅列作工程记录。

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
| my-gpu | class_independent_spatial_sweep_20260908/jobs/cl_spatial_0.01 | 3 | 40 | 含背景分项缺失 | NA | S088 |
| my-gpu | class_independent_spatial_sweep_20260908/jobs/ind_T1_lr0.01 | 1 | 20 | 含背景分项缺失 | NA | S089 |
| my-gpu | class_independent_spatial_sweep_20260908/jobs/ind_T1_lr0.03 | 1 | 20 | 含背景分项缺失 | NA | S090 |
| my-gpu | class_independent_spatial_sweep_20260908/jobs/ind_T2_lr0.01 | 1 | 20 | 含背景分项缺失 | NA | S091 |
| my-gpu | class_independent_spatial_sweep_20260908/jobs/ind_T2_lr0.03 | 1 | 20 | 含背景分项缺失 | NA | S092 |
| my-gpu | class_independent_spatial_sweep_20260908/jobs/ind_T3_lr0.01 | 1 | 20 | 含背景分项缺失 | NA | S093 |
| my-gpu | class_independent_spatial_sweep_20260908/jobs/ind_T3_lr0.03 | 1 | 20 | 含背景分项缺失 | NA | S094 |
| my-gpu | class_independent_spatial_sweep_20260908/smoke_cl_spatial_original_gco | 3 | 1 | 工程/短诊断；含背景分项缺失 | NA | S095 |
| my-gpu | class_independent_spatial_sweep_20260908/smoke_ind_T3_gco | 1 | 1 | 工程/短诊断；含背景分项缺失 | NA | S096 |
| my-gpu | class_replay_improve_20260915/jobs/zs-der-smoke | 2 | 80/1/60 | 工程/短诊断；仅当前任务验证 | 0.324897 | S097 |
| my-gpu | class_replay_improve_20260915_r2/jobs/zs-der-smoke | 2 | 80/1/60 | 工程/短诊断；仅当前任务验证 | 0.324897 | S098 |
| my-gpu | class_replay_improve_20260915_r2/jobs/zs-er-smoke | 2 | 80/1/60 | 工程/短诊断；仅当前任务验证 | 0.318258 | S099 |
| my-gpu | core_runs/class_q8v2n6_seed42_150e_20260811T1345Z/pce_mib_seed42 | 3 | 150 | 含背景分项缺失 | NA | S100 |
| my-gpu | core_runs/class_q8v2n6_seed42_150e_20260811T1345Z/pce_sequential_seed42 | 3 | 150 | 含背景分项缺失 | NA | S101 |
| my-gpu | core_runs/class_q8v2n6_seed42_150e_20260811T1345Z/zs_mib_global1_origscale_seed42 | 3 | 150 | 含背景分项缺失 | NA | S102 |
| my-gpu | core_runs/class_q8v2n6_seed42_150e_20260811T1345Z/zs_sequential_global1_origscale_seed42 | 3 | 150 | 含背景分项缺失 | NA | S103 |
| my-gpu | domain_joint_runs_20260901/runs/j0b2l15_r1 | — | 150 | 含背景分项缺失 | NA | S104 |
| my-gpu | domain_joint_runs_20260901/runs/j1b4l30_r1 | — | 150 | 含背景分项缺失 | NA | S105 |
| my-gpu | domain_runs_20260901/runs/d0gpm_r1 | 6 | 150 | 含背景分项缺失 | NA | S106 |
| my-gpu | domain_runs_20260901/runs/d3p9n_r1 | 6 | 150 | 含背景分项缺失 | NA | S107 |
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

| run_id | global_weight | buffer_size | minibatch_size | alpha | beta | status | audit | 含背景 Dice |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| a2m7x | 1.0 | 64 | 8 | 0.25 | 0.5 | complete | PASS | NA |
| c4n9v | 1.0 | 64 | 8 | 0.5 | 0.5 | complete | PASS | NA |
| e6p3r | 1.0 | 64 | 8 | 0.5 | 1.0 | complete | PASS | NA |
| g8t5w | 1.0 | 64 | 8 | 1.0 | 1.0 | complete | PASS | NA |
| b3k8p | 0.1 | 64 | 8 | 1.0 | 1.0 | complete | PASS | NA |
| d5m2v | 1.0 | 128 | 8 | 1.0 | 1.0 | complete | PASS | NA |
| f7r4x | 1.0 | 64 | 16 | 1.0 | 1.0 | NA | NA | NA |
| h9t6z | 1.0 | 64 | 8 | 1.0 | 0.5 | complete | PASS | NA |

**organ_zs_derpp_ab_sweep_20260902**（S140）

| run_id | global_weight | buffer_size | minibatch_size | alpha | beta | audit | 含背景 Dice |
| --- | --- | --- | --- | --- | --- | --- | --- |
| j3p7n | 1.0 | 64 | 8 | 0.5 | 0.5 | PASS | NA |
| k5r9v | 1.0 | 64 | 8 | 1.0 | 0.5 | PASS | NA |
| l7t4x | 1.0 | 64 | 8 | 1.0 | 1.0 | PASS | NA |
| m2v6q | 1.0 | 64 | 8 | 0.25 | 0.5 | PASS | NA |
| n4x8r | 1.0 | 64 | 8 | 0.5 | 1.0 | PASS | NA |
| p6z3t | 0.1 | 64 | 8 | 0.5 | 0.5 | PASS | NA |

**joint_short_20260902**（S141）

| run_id | learning_rate | epochs | audit | 含背景 Dice |
| --- | --- | --- | --- | --- |
| h4m8q | .03 | 1 | PASS | NA |
| k7v2n | .06 | 1 | PASS | NA |
| p3x6d | .10 | 1 | PASS | NA |
| r8c4w | .02 | 3 | PASS | NA |
| t5n9b | .03 | 3 | PASS | NA |
| u2f7k | .04 | 3 | PASS | NA |
| w6d3s | .03 | 5 | PASS | NA |
| y9h4m | .04 | 5 | PASS | 0.776010 |

Joint 的 y9h4m 已在第 4 节由独立的背景复评文件补齐；其余 Joint 及两类 replay 参数筛选仍缺背景分项。Domain 参数筛选中 f7r4x 的 minibatch=16 超出当时 24 GB 显存条件，属于不可行候选，不是完成的零分实验。

### 7.2 更早的静态参考、覆盖率与过拟合诊断

静态参考表含显式病例级含背景字段，保留如下。它是历史 checkpoint 导出结果，当前未重新确认其数据划分，不能并入正式测试主表。

| run | 变体 | 记录状态 | best checkpoint epoch（原始索引） | best 含背景 Dice | last 含背景 Dice | 来源 |
| --- | --- | --- | --- | --- | --- | --- |
| static_A0_sgd_seed42 | fg_only | completed | 4 | 0.146502 | 0.105977 | S142 |
| static_A_ratio_sgd_seed42 | legacy_ratio | completed | 10 | 0.509141 | 0.373760 | S142 |
| static_Dense_v2_sgd_seed42 | dense | completed | 136 | 0.818302 | 0.811501 | S142 |

| run | 记录任务 | 指标限制 | 含背景 Dice | 来源 |
| --- | --- | --- | --- | --- |
| coverage_runs/B1/pce_seed42_stage1 | 1 | 历史阶段指标；背景口径未确认 | NA | S143 |
| coverage_runs/B1/zs_seed42_stage1 | 1 | 历史阶段指标；背景口径未确认 | NA | S144 |
| coverage_runs/B2/pce_seed42_stage1 | 1 | 历史阶段指标；背景口径未确认 | NA | S145 |
| coverage_runs/B2/zs_seed42_stage1 | 1 | 历史阶段指标；背景口径未确认 | NA | S146 |
| coverage_runs/B3/pce_seed42_stage1 | 1 | 历史阶段指标；背景口径未确认 | NA | S147 |
| coverage_runs/B3/zs_seed42_stage1 | 1 | 历史阶段指标；背景口径未确认 | NA | S148 |
| coverage_runs/Dense/dense_seed42_stage1 | 1 | 历史阶段指标；背景口径未确认 | NA | S149 |
| overfit/pce_mib_seed42_stage2 | 1,2 | 历史阶段指标；背景口径未确认 | NA | S150 |
| overfit/pce_seed42_stage1 | 1 | 历史阶段指标；背景口径未确认 | NA | S151 |
| overfit/zs_mib_seed42_stage2 | 1,2 | 历史阶段指标；背景口径未确认 | NA | S152 |
| overfit/zs_seed42_stage1 | 1 | 历史阶段指标；背景口径未确认 | NA | S153 |
| overfit_fixed/pce_mib_seed42_stage2 | 1,2 | 历史阶段指标；背景口径未确认 | NA | S154 |

## 8. 无完整 summary 的产物索引

以下目录有 manifest，但未找到对应完整 summary。这里只登记记录状态，不把 manifest 里历史遗留的 `running` 解释为当前仍有训练进程。早期独立训练已由第 6 节 CSV 完整覆盖的目录、resume 前缀拷贝和参考镜像不重复列出。需要数值补齐时，应从保留 checkpoint 与同一评估协议复评；本次没有用猜测补值。

| 位置 | 批次 / run | 证据状态 | 配置预算 | 完整含背景 Dice | 来源 |
| --- | --- | --- | --- | --- | --- |
| my-gpu | class_comparisons_20260910/jobs/pce_sequential | 无最终 summary | 80 | NA | S155 |
| my-gpu | class_comparisons_20260910/jobs/zs_sequential | 无最终 summary | 80 | NA | S156 |
| my-gpu | class_er_gpu3_20260913/jobs/zs-er | 无最终 summary | 80 | NA | S157 |
| my-gpu | class_independent_spatial_sweep_20260908/jobs/cl_spatial_0.001 | 无最终 summary | 40 | NA | S158 |
| my-gpu | class_independent_spatial_sweep_20260908/jobs/ind_T1_formal | 无最终 summary | 80 | NA | S159 |
| my-gpu | class_independent_spatial_sweep_20260908/smoke_cl_spatial_original | 无最终 summary | 1 | NA | S160 |
| my-gpu | class_independent_spatial_sweep_20260908/smoke_ind_T3 | 无最终 summary | 1 | NA | S161 |
| my-gpu | class_replay_comparisons_20260912/jobs/zs-der | 无最终 summary | 80 | NA | S162 |
| my-gpu | class_replay_comparisons_20260912/jobs/zs-er | 无最终 summary | 80 | NA | S163 |
| my-gpu | class_replay_improve_20260915/jobs/zs-der-control5 | 无最终 summary | 80/5/60 | NA | S164 |
| my-gpu | class_replay_improve_20260915/jobs/zs-er-smoke | 无最终 summary | 80/1/60 | NA | S165 |
| my-gpu | class_replay_resume_20260913/jobs/zs-der | 无最终 summary | 80 | NA | S166 |
| my-gpu | class_replay_resume_20260913/jobs/zs-er | 无最终 summary | 80 | NA | S167 |
| my-gpu | core_runs/class_q8v2n6_seed42_150e_20260811T1345Z/zs_mib_raw_global1_diagnostic_stopped_iter1000_seed42 | 无最终 summary | 150 | NA | S168 |
| my-gpu | core_runs/class_q8v2n6_seed42_150e_20260811T1345Z/zs_sequential_raw_global1_diagnostic_stopped_iter1200_seed42 | 无最终 summary | 150 | NA | S169 |
| my-gpu | domain_joint_c3fix_20260902/runs/c3adam_e5_s42 | 历史 manifest 标 running；未确认存活 | 5 | NA | S170 |
| my-gpu | domain_joint_c3fix_20260902/runs/c3adamgd_e5_s42 | 历史 manifest 标 running；未确认存活 | 5 | NA | S171 |
| my-gpu | domain_joint_c3fix_20260902/runs/c3sgd_e3_s42 | 历史 manifest 标 running；未确认存活 | 3 | NA | S172 |
| my-gpu | domain_joint_validation_20260902/runs/balpce_b4e20_s42 | 历史 manifest 标 running；未确认存活 | 20 | NA | S173 |
| my-gpu | domain_joint_validation_20260902/runs/balzs_b4e20_s42 | 历史 manifest 标 running；未确认存活 | 20 | NA | S174 |
| my-gpu | organ_runs_20260901/runs/o1seq_r1 | 无最终 summary | 150 | NA | S175 |
| my-gpu | organ_runs_20260901/runs/o2ewc_r1 | 无最终 summary | 150 | NA | S176 |
| my-gpu | organ_runs_20260901/runs/u2k7m | 无最终 summary | 150 | NA | S177 |
| my-gpu | organ_runs_20260901/runs/u2k7m_r1 | 无最终 summary | 150 | NA | S178 |
| my-gpu | organ_runs_20260901/runs/u2k7m_r2 | 无最终 summary | 150 | NA | S179 |
| my-gpu | organ_runs_20260901/runs/v3p8n | 无最终 summary | 150 | NA | S180 |
| my-gpu | organ_runs_20260901/runs/v3p8n_r1 | 无最终 summary | 150 | NA | S181 |
| my-gpu | organ_runs_20260901/runs/v3p8n_r2 | 无最终 summary | 150 | NA | S182 |
| jiangsuiyang | class_independent_spatial_sweep_20260908/jobs/ind_T1_lr0.01 | 无最终 summary | 20 | NA | S183 |
| jiangsuiyang | class_independent_spatial_sweep_20260908/jobs/ind_T2_lr0.01 | 无最终 summary | 20 | NA | S184 |
| jiangsuiyang | class_independent_spatial_sweep_20260908/jobs/ind_T3_lr0.01 | 无最终 summary | 20 | NA | S185 |
| jiangsuiyang | class_independent_spatial_sweep_20260908/smoke_cl_spatial | 无最终 summary | 1 | NA | S186 |
| jiangsuiyang | independent80_seed42_20260906/smoke_class_T2_scribble | complete | 1 | NA | S187 |
| jiangsuiyang | independent80_seed42_20260906/smoke_domain_A_full | complete | 1 | NA | S188 |
| jiangsuiyang | organ_CL_throughput_20260908/spatial_off/run | 无最终 summary | 2 | NA | S189 |
| jiangsuiyang | organ_T13_half_cl_20260908/balance_checks_20260908_v2/balanced_current_full_replay/run | 无最终 summary | 60 | NA | S190 |
| jiangsuiyang | organ_T13_half_cl_20260908/balance_checks_20260908_v2/feature_replay_only/run | 无最终 summary | 60 | NA | S191 |
| jiangsuiyang | organ_T13_half_cl_20260908/balance_checks_20260908_v2/no_replay_losses/run | 无最终 summary | 60 | NA | S192 |
| jiangsuiyang | organ_T13_half_cl_20260908/balance_checks_20260908_v2/supervision_replay_only/run | 无最终 summary | 60 | NA | S193 |
| jiangsuiyang | organ_T13_half_cl_20260908/diagnostic_t2_seed43_v3/run | 无最终 summary | 60 | NA | S194 |
| jiangsuiyang | organ_T13_half_cl_20260908/formal_controls_20260908/alpha0 | 无最终 summary | 60 | NA | S195 |
| jiangsuiyang | organ_T13_half_cl_20260908/formal_controls_20260908/alpha01 | 无最终 summary | 60 | NA | S196 |
| jiangsuiyang | organ_T13_half_cl_20260908/formal_controls_20260908/no_spatial | 无最终 summary | 60 | NA | S197 |
| jiangsuiyang | organ_T13_half_cl_20260908/run | 无最终 summary | 80 | NA | S198 |
| jiangsuiyang | organ_T13_half_cl_20260908/run60 | 无最终 summary | 60 | NA | S199 |
| jiangsuiyang | organ_T13_half_cl_20260908/short_checks_20260908/clean_bn/run | 无最终 summary | 60 | NA | S200 |
| jiangsuiyang | organ_T13_half_cl_20260908/short_checks_20260908/clean_bn_clip5_126/run | 无最终 summary | 60 | NA | S201 |
| jiangsuiyang | organ_T13_half_cl_20260908/short_checks_20260908/clip5/run | 无最终 summary | 60 | NA | S202 |
| jiangsuiyang | organ_T13_half_cl_20260908/short_checks_20260908/clip5_126/run | 无最终 summary | 60 | NA | S203 |
| jiangsuiyang | organ_T13_half_cl_20260908/short_checks_20260908/clip5_420/run | 无最终 summary | 60 | NA | S204 |
| jiangsuiyang | organ_T13_half_cl_20260908/short_checks_20260908/clip5_seed44_126/run | 无最终 summary | 60 | NA | S205 |
| jiangsuiyang | organ_T13_half_cl_20260908/short_checks_20260908/frozen_bn_126/run | 无最终 summary | 60 | NA | S206 |
| jiangsuiyang | organ_T13_half_cl_20260908/short_checks_20260908/frozen_bn_clip5_126/run | 无最终 summary | 60 | NA | S207 |
| jiangsuiyang | organ_T13_half_cl_20260908/short_checks_20260908/lr003/run | 无最终 summary | 60 | NA | S208 |
| jiangsuiyang | organ_T13_half_cl_20260908/short_checks_20260908/lr003_126/run | 无最终 summary | 60 | NA | S209 |
| jiangsuiyang | organ_T13_half_cl_20260908/short_checks_20260908/lr003_clip5/run | 无最终 summary | 60 | NA | S210 |
| jiangsuiyang | organ_T13_half_cl_20260908/short_checks_20260908/lr003_clip5_126/run | 无最终 summary | 60 | NA | S211 |
| jiangsuiyang | organ_T13_half_cl_20260908/small_alpha_checks_20260908/alpha_0.01/run | 无最终 summary | 60 | NA | S212 |
| jiangsuiyang | organ_T13_half_cl_20260908/small_alpha_checks_20260908/alpha_0.05/run | 无最终 summary | 60 | NA | S213 |
| jiangsuiyang | organ_T13_half_cl_20260908/small_alpha_checks_20260908/alpha_0.1/run | 无最终 summary | 60 | NA | S214 |
| jiangsuiyang | organ_T2_coverage_20260908/runs/cl_fg20 | 无最终 summary | 80 | NA | S215 |
| jiangsuiyang | organ_T2_coverage_20260908/runs/cl_fg40 | 无最终 summary | 80 | NA | S216 |
| jiangsuiyang | organ_T2_coverage_20260908/runs/stability_gate | 无最终 summary | 2 | NA | S217 |
| jiangsuiyang | organ_T2_coverage_20260908/runs/sw_lr01_g0 | 无最终 summary | 20 | NA | S218 |
| jiangsuiyang | organ_T2_coverage_20260908/runs/sw_lr01_g01 | 无最终 summary | 20 | NA | S219 |
| jiangsuiyang | organ_T2_coverage_20260908/runs/sw_lr03_g0 | 无最终 summary | 20 | NA | S220 |
| jiangsuiyang | organ_T2_coverage_20260908/runs/sw_lr03_g01 | 无最终 summary | 20 | NA | S221 |
| jiangsuiyang | organ_T34_lr006_spatial_pair20_20260910/runs/spatial0 | 无最终 summary | 20 | NA | S222 |
| jiangsuiyang | organ_T34_lr006_spatial_pair20_20260910/runs/spatial001 | 无最终 summary | 20 | NA | S223 |
| jiangsuiyang | organ_batch8_probe_20260908/b8_spatial_on/run | 无最终 summary | 1 | NA | S224 |
| jiangsuiyang | organ_metrics_half_20260909/prefix_T2 | 无最终 summary | 60 | NA | S225 |
| jiangsuiyang | organ_metrics_half_20260909/runs/ind_T4 | 无最终 summary | 80 | NA | S226 |
| jiangsuiyang | organ_t3_retention_probe_20260909/R0 | 无最终 summary | 60 | NA | S227 |
| jiangsuiyang | organ_t3_retention_probe_20260909/R1 | 无最终 summary | 60 | NA | S228 |
| jiangsuiyang | organ_t3_retention_probe_20260909/R2 | 无最终 summary | 60 | NA | S229 |
| jiangsuiyang | organ_t3_retention_probe_20260909/R3 | 无最终 summary | 60 | NA | S230 |
| jiangsuiyang | organ_t3_retention_probe_20260909/frozen | 无最终 summary | 60 | NA | S231 |
| jiangsuiyang | replay_comparisons_20260911/runs/domain_zs-der | 无最终 summary | 80 | NA | S232 |
| jiangsuiyang | replay_comparisons_20260911/runs/domain_zs-er | 无最终 summary | 80 | NA | S233 |
| jiangsuiyang | replay_comparisons_20260911/runs/organ_zs-der | 无最终 summary | 40 | NA | S234 |
| jiangsuiyang | tune_independent_A_07261_seed42_20260907_1303/independent | stopped_protocol_mismatch | 150 | NA | S235 |
| jiangsuiyang | tune_independent_A_07261_seed42_20260907_1303/oracle | stopped_protocol_mismatch | 150 | NA | S236 |
| jiangsuiyang | tune_independent_A_07261_seed42_20260907_1303/parity/independent | 历史 manifest 标 running；未确认存活 | 150 | NA | S237 |
| jiangsuiyang | tune_independent_A_07261_seed42_20260907_1303/parity/oracle | 历史 manifest 标 running；未确认存活 | 150 | NA | S238 |
| jiangsuiyang | tune_independent_A_07261_seed42_20260907_1303/parity_deterministic/independent | 历史 manifest 标 running；未确认存活 | 150 | NA | S239 |
| jiangsuiyang | tune_independent_A_07261_seed42_20260907_1303/parity_deterministic/oracle | 历史 manifest 标 running；未确认存活 | 150 | NA | S240 |
| jiangsuiyang | tune_independent_A_07261_seed42_20260907_1303/parity_pattern_f5_b10/independent | 历史 manifest 标 running；未确认存活 | 150 | NA | S241 |
| jiangsuiyang | tune_independent_A_07261_seed42_20260907_1303/parity_pattern_f5_b10/oracle | 历史 manifest 标 running；未确认存活 | 150 | NA | S242 |
| jiangsuiyang | tune_independent_A_07261_seed42_20260907_1303/parity_pattern_f5_b10/oracle_repeat | 历史 manifest 标 running；未确认存活 | 150 | NA | S243 |
| jiangsuiyang | tune_independent_A_07261_seed42_20260907_1303/parity_repeat/independent | 历史 manifest 标 running；未确认存活 | 150 | NA | S244 |
| jiangsuiyang | tune_independent_A_07261_seed42_20260907_1303/parity_repeat/oracle | 历史 manifest 标 running；未确认存活 | 150 | NA | S245 |
| jiangsuiyang | tune_independent_A_07261_seed42_20260907_1303/parity_repeat/oracle_repeat | 历史 manifest 标 running；未确认存活 | 150 | NA | S246 |
| jiangsuiyang | zs_independent_search_seed42_20260906_inclusive_aborted_20260906_2128/tuning/class_T2_c01 | 历史 manifest 标 running；未确认存活 | 20 | NA | S247 |
| jiangsuiyang | zs_independent_search_seed42_20260906_inclusive_aborted_20260906_2128/tuning/class_T2_c02 | 历史 manifest 标 running；未确认存活 | 20 | NA | S248 |
| jiangsuiyang | zs_independent_search_seed42_20260906_inclusive_aborted_20260906_2128/tuning/class_T2_c03 | 历史 manifest 标 running；未确认存活 | 20 | NA | S249 |
| jiangsuiyang | zs_independent_search_seed42_20260906_inclusive_aborted_20260906_2128/tuning/class_T2_c04 | 历史 manifest 标 running；未确认存活 | 20 | NA | S250 |
| jiangsuiyang | zs_independent_search_seed42_20260906_inclusive_aborted_20260906_2128/tuning/domain_C_c01 | 历史 manifest 标 running；未确认存活 | 20 | NA | S251 |
| jiangsuiyang | zs_independent_search_seed42_20260906_inclusive_aborted_20260906_2128/tuning/domain_C_c02 | 历史 manifest 标 running；未确认存活 | 20 | NA | S252 |
| jiangsuiyang | zs_independent_search_seed42_20260906_inclusive_aborted_20260906_2128/tuning/domain_C_c03 | 历史 manifest 标 running；未确认存活 | 20 | NA | S253 |
| jiangsuiyang | zs_independent_search_seed42_20260906_inclusive_aborted_20260906_2128/tuning/domain_C_c04 | 历史 manifest 标 running；未确认存活 | 20 | NA | S254 |
| jiangsuiyang | zs_independent_tuning_seed42_20260906/smoke_class_T2 | complete | 2 | NA | S255 |
| jiangsuiyang | zs_independent_tuning_seed42_20260906/smoke_domain_B_skiptest | complete | 1 | NA | S256 |

## 9. 结果使用结论与缺项

1. 当前同预算 Class 正式组已具备完整含背景结果、阶段矩阵与统一独立参考；ZS-DER++-MiB 的 A-Dice=0.767664。ER/guarded DER 已完成，但训练预算不同且旧任务结果弱。
2. Organ ER 已完成且含背景 A-Dice=0.784476。最新主方法与多数 Organ baseline 的原始汇总只保存前景分数，不能据此给出含背景统一排名。u5k2n 的 0.768314 来自未完成 checkpoint。
3. 归档 Domain DER++ 为 0.768733，GPM 为 0.657588，但预算分别为 80 与 150 epochs/任务。新 Domain ER/DER 失败，不能用缺失结果代表 0 分或宣称完成。
4. 独立训练、验证筛选、测试选模演示、短诊断与正式持续学习分别展示。所有缺项均保留，不用背景≈1 的假设估算，不复制前景 BWTR/RMA 冒充含背景结果。
5. 后续补齐仅需对缺背景分项的已有 checkpoint 按原数据划分复评，并保存每任务背景及各类别指标、阶段矩阵和选模来源。该复评不属于本次已有结果整理，尚未执行。

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
| S015 | jiangsuiyang | `organ_T3_lr006_spatial_pair10_20260910/runs/spatial0/summary.json` |
| S016 | jiangsuiyang | `organ_T3_lr006_spatial_pair10_20260910/runs/spatial001/summary.json` |
| S017 | jiangsuiyang | `organ_T4_from_T3best_lr006_spatial_pair10_20260910/runs/spatial0/summary.json` |
| S018 | jiangsuiyang | `organ_T4_from_T3best_lr006_spatial_pair10_20260910/runs/spatial001/summary.json` |
| S019 | jiangsuiyang | `organ_baseline_tuning_20260911/runs/zs-ewc_c0/summary.json` |
| S020 | jiangsuiyang | `organ_baseline_tuning_20260911/runs/zs-ewc_c1/summary.json` |
| S021 | jiangsuiyang | `organ_baseline_tuning_20260911/runs/zs-ewc_c2/summary.json` |
| S022 | jiangsuiyang | `organ_baseline_tuning_20260911/runs/zs-ewc_c3/summary.json` |
| S023 | jiangsuiyang | `organ_baseline_tuning_20260911/runs/zs-ewc_formal/summary.json` |
| S024 | jiangsuiyang | `organ_baseline_tuning_20260911/runs/zs-gpm_c0/summary.json` |
| S025 | jiangsuiyang | `organ_baseline_tuning_20260911/runs/zs-gpm_c1/summary.json` |
| S026 | jiangsuiyang | `organ_baseline_tuning_20260911/runs/zs-gpm_c2/summary.json` |
| S027 | jiangsuiyang | `organ_baseline_tuning_20260911/runs/zs-gpm_c3/summary.json` |
| S028 | jiangsuiyang | `organ_baseline_tuning_20260911/runs/zs-gpm_formal/summary.json` |
| S029 | jiangsuiyang | `organ_baseline_tuning_20260911/runs/zs-sequential_c0/summary.json` |
| S030 | jiangsuiyang | `organ_baseline_tuning_20260911/runs/zs-sequential_c1/summary.json` |
| S031 | jiangsuiyang | `organ_baseline_tuning_20260911/runs/zs-sequential_c2/summary.json` |
| S032 | jiangsuiyang | `organ_baseline_tuning_20260911/runs/zs-sequential_c3/summary.json` |
| S033 | jiangsuiyang | `organ_baseline_tuning_20260911/runs/zs-sequential_formal/summary.json` |
| S034 | jiangsuiyang | `organ_comparisons_20260910/runs/dense-sequential/summary.json` |
| S035 | jiangsuiyang | `organ_comparisons_20260910/runs/pce-sequential/summary.json` |
| S036 | jiangsuiyang | `organ_comparisons_20260910/runs/zs-ewc/summary.json` |
| S037 | jiangsuiyang | `organ_comparisons_20260910/runs/zs-gpm/summary.json` |
| S038 | jiangsuiyang | `organ_comparisons_20260910/runs/zs-sequential/summary.json` |
| S039 | jiangsuiyang | `organ_metrics_T134_half_20260909/runs/ind_T4/summary.json` |
| S040 | jiangsuiyang | `organ_metrics_half_20260909/runs/ind_T3/summary.json` |
| S041 | jiangsuiyang | `organ_metrics_half_20260909/runs/organ_no_replay/summary.json` |
| S042 | jiangsuiyang | `organ_metrics_half_20260909/runs/organ_retention/summary.json` |
| S043 | jiangsuiyang | `replay_comparisons_20260911/progress.json` |
| S044 | local | `independent_a_work/results/domain_zs_derpp/background_comparison.json` |
| S045 | local | `independent_a_work/results/domain_zs_gpm/background_comparison.json` |
| S046 | local | `independent_a_work/results/joint_short_20260902/background_comparison.json` |
| S047 | my-gpu | `class_replay_improve_20260915_r2/progress.json` |
| S048 | my-gpu | `class_replay_improve_20260915_r2/jobs/zs-er-control5/summary.json` |
| S049 | my-gpu | `class_replay_improve_20260915_r2/jobs/zs-er-improved5/summary.json` |
| S050 | my-gpu | `class_replay_improve_20260915_r2/jobs/zs-der-control5/summary.json` |
| S051 | my-gpu | `class_replay_improve_20260915_r2/jobs/zs-der-improved5/summary.json` |
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
| S088 | my-gpu | `class_independent_spatial_sweep_20260908/jobs/cl_spatial_0.01/summary.json` |
| S089 | my-gpu | `class_independent_spatial_sweep_20260908/jobs/ind_T1_lr0.01/summary.json` |
| S090 | my-gpu | `class_independent_spatial_sweep_20260908/jobs/ind_T1_lr0.03/summary.json` |
| S091 | my-gpu | `class_independent_spatial_sweep_20260908/jobs/ind_T2_lr0.01/summary.json` |
| S092 | my-gpu | `class_independent_spatial_sweep_20260908/jobs/ind_T2_lr0.03/summary.json` |
| S093 | my-gpu | `class_independent_spatial_sweep_20260908/jobs/ind_T3_lr0.01/summary.json` |
| S094 | my-gpu | `class_independent_spatial_sweep_20260908/jobs/ind_T3_lr0.03/summary.json` |
| S095 | my-gpu | `class_independent_spatial_sweep_20260908/smoke_cl_spatial_original_gco/summary.json` |
| S096 | my-gpu | `class_independent_spatial_sweep_20260908/smoke_ind_T3_gco/summary.json` |
| S097 | my-gpu | `class_replay_improve_20260915/jobs/zs-der-smoke/summary.json` |
| S098 | my-gpu | `class_replay_improve_20260915_r2/jobs/zs-der-smoke/summary.json` |
| S099 | my-gpu | `class_replay_improve_20260915_r2/jobs/zs-er-smoke/summary.json` |
| S100 | my-gpu | `core_runs/class_q8v2n6_seed42_150e_20260811T1345Z/pce_mib_seed42/summary.json` |
| S101 | my-gpu | `core_runs/class_q8v2n6_seed42_150e_20260811T1345Z/pce_sequential_seed42/summary.json` |
| S102 | my-gpu | `core_runs/class_q8v2n6_seed42_150e_20260811T1345Z/zs_mib_global1_origscale_seed42/summary.json` |
| S103 | my-gpu | `core_runs/class_q8v2n6_seed42_150e_20260811T1345Z/zs_sequential_global1_origscale_seed42/summary.json` |
| S104 | my-gpu | `domain_joint_runs_20260901/runs/j0b2l15_r1/summary.json` |
| S105 | my-gpu | `domain_joint_runs_20260901/runs/j1b4l30_r1/summary.json` |
| S106 | my-gpu | `domain_runs_20260901/runs/d0gpm_r1/summary.json` |
| S107 | my-gpu | `domain_runs_20260901/runs/d3p9n_r1/summary.json` |
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
| S142 | my-gpu | `static_reference_exports_20260807/static_reference_results.csv` |
| S143 | my-gpu | `coverage_runs/B1/pce_seed42_stage1/stage_metrics.csv` |
| S144 | my-gpu | `coverage_runs/B1/zs_seed42_stage1/stage_metrics.csv` |
| S145 | my-gpu | `coverage_runs/B2/pce_seed42_stage1/stage_metrics.csv` |
| S146 | my-gpu | `coverage_runs/B2/zs_seed42_stage1/stage_metrics.csv` |
| S147 | my-gpu | `coverage_runs/B3/pce_seed42_stage1/stage_metrics.csv` |
| S148 | my-gpu | `coverage_runs/B3/zs_seed42_stage1/stage_metrics.csv` |
| S149 | my-gpu | `coverage_runs/Dense/dense_seed42_stage1/stage_metrics.csv` |
| S150 | my-gpu | `overfit/pce_mib_seed42_stage2/stage_metrics.csv` |
| S151 | my-gpu | `overfit/pce_seed42_stage1/stage_metrics.csv` |
| S152 | my-gpu | `overfit/zs_mib_seed42_stage2/stage_metrics.csv` |
| S153 | my-gpu | `overfit/zs_seed42_stage1/stage_metrics.csv` |
| S154 | my-gpu | `overfit_fixed/pce_mib_seed42_stage2/stage_metrics.csv` |
| S155 | my-gpu | `class_comparisons_20260910/jobs/pce_sequential/manifest.json` |
| S156 | my-gpu | `class_comparisons_20260910/jobs/zs_sequential/manifest.json` |
| S157 | my-gpu | `class_er_gpu3_20260913/jobs/zs-er/manifest.json` |
| S158 | my-gpu | `class_independent_spatial_sweep_20260908/jobs/cl_spatial_0.001/manifest.json` |
| S159 | my-gpu | `class_independent_spatial_sweep_20260908/jobs/ind_T1_formal/manifest.json` |
| S160 | my-gpu | `class_independent_spatial_sweep_20260908/smoke_cl_spatial_original/manifest.json` |
| S161 | my-gpu | `class_independent_spatial_sweep_20260908/smoke_ind_T3/manifest.json` |
| S162 | my-gpu | `class_replay_comparisons_20260912/jobs/zs-der/manifest.json` |
| S163 | my-gpu | `class_replay_comparisons_20260912/jobs/zs-er/manifest.json` |
| S164 | my-gpu | `class_replay_improve_20260915/jobs/zs-der-control5/manifest.json` |
| S165 | my-gpu | `class_replay_improve_20260915/jobs/zs-er-smoke/manifest.json` |
| S166 | my-gpu | `class_replay_resume_20260913/jobs/zs-der/manifest.json` |
| S167 | my-gpu | `class_replay_resume_20260913/jobs/zs-er/manifest.json` |
| S168 | my-gpu | `core_runs/class_q8v2n6_seed42_150e_20260811T1345Z/zs_mib_raw_global1_diagnostic_stopped_iter1000_seed42/manifest.json` |
| S169 | my-gpu | `core_runs/class_q8v2n6_seed42_150e_20260811T1345Z/zs_sequential_raw_global1_diagnostic_stopped_iter1200_seed42/manifest.json` |
| S170 | my-gpu | `domain_joint_c3fix_20260902/runs/c3adam_e5_s42/manifest.json` |
| S171 | my-gpu | `domain_joint_c3fix_20260902/runs/c3adamgd_e5_s42/manifest.json` |
| S172 | my-gpu | `domain_joint_c3fix_20260902/runs/c3sgd_e3_s42/manifest.json` |
| S173 | my-gpu | `domain_joint_validation_20260902/runs/balpce_b4e20_s42/manifest.json` |
| S174 | my-gpu | `domain_joint_validation_20260902/runs/balzs_b4e20_s42/manifest.json` |
| S175 | my-gpu | `organ_runs_20260901/runs/o1seq_r1/manifest.json` |
| S176 | my-gpu | `organ_runs_20260901/runs/o2ewc_r1/manifest.json` |
| S177 | my-gpu | `organ_runs_20260901/runs/u2k7m/manifest.json` |
| S178 | my-gpu | `organ_runs_20260901/runs/u2k7m_r1/manifest.json` |
| S179 | my-gpu | `organ_runs_20260901/runs/u2k7m_r2/manifest.json` |
| S180 | my-gpu | `organ_runs_20260901/runs/v3p8n/manifest.json` |
| S181 | my-gpu | `organ_runs_20260901/runs/v3p8n_r1/manifest.json` |
| S182 | my-gpu | `organ_runs_20260901/runs/v3p8n_r2/manifest.json` |
| S183 | jiangsuiyang | `class_independent_spatial_sweep_20260908/jobs/ind_T1_lr0.01/manifest.json` |
| S184 | jiangsuiyang | `class_independent_spatial_sweep_20260908/jobs/ind_T2_lr0.01/manifest.json` |
| S185 | jiangsuiyang | `class_independent_spatial_sweep_20260908/jobs/ind_T3_lr0.01/manifest.json` |
| S186 | jiangsuiyang | `class_independent_spatial_sweep_20260908/smoke_cl_spatial/manifest.json` |
| S187 | jiangsuiyang | `independent80_seed42_20260906/smoke_class_T2_scribble/manifest.json` |
| S188 | jiangsuiyang | `independent80_seed42_20260906/smoke_domain_A_full/manifest.json` |
| S189 | jiangsuiyang | `organ_CL_throughput_20260908/spatial_off/run/manifest.json` |
| S190 | jiangsuiyang | `organ_T13_half_cl_20260908/balance_checks_20260908_v2/balanced_current_full_replay/run/manifest.json` |
| S191 | jiangsuiyang | `organ_T13_half_cl_20260908/balance_checks_20260908_v2/feature_replay_only/run/manifest.json` |
| S192 | jiangsuiyang | `organ_T13_half_cl_20260908/balance_checks_20260908_v2/no_replay_losses/run/manifest.json` |
| S193 | jiangsuiyang | `organ_T13_half_cl_20260908/balance_checks_20260908_v2/supervision_replay_only/run/manifest.json` |
| S194 | jiangsuiyang | `organ_T13_half_cl_20260908/diagnostic_t2_seed43_v3/run/manifest.json` |
| S195 | jiangsuiyang | `organ_T13_half_cl_20260908/formal_controls_20260908/alpha0/manifest.json` |
| S196 | jiangsuiyang | `organ_T13_half_cl_20260908/formal_controls_20260908/alpha01/manifest.json` |
| S197 | jiangsuiyang | `organ_T13_half_cl_20260908/formal_controls_20260908/no_spatial/manifest.json` |
| S198 | jiangsuiyang | `organ_T13_half_cl_20260908/run/manifest.json` |
| S199 | jiangsuiyang | `organ_T13_half_cl_20260908/run60/manifest.json` |
| S200 | jiangsuiyang | `organ_T13_half_cl_20260908/short_checks_20260908/clean_bn/run/manifest.json` |
| S201 | jiangsuiyang | `organ_T13_half_cl_20260908/short_checks_20260908/clean_bn_clip5_126/run/manifest.json` |
| S202 | jiangsuiyang | `organ_T13_half_cl_20260908/short_checks_20260908/clip5/run/manifest.json` |
| S203 | jiangsuiyang | `organ_T13_half_cl_20260908/short_checks_20260908/clip5_126/run/manifest.json` |
| S204 | jiangsuiyang | `organ_T13_half_cl_20260908/short_checks_20260908/clip5_420/run/manifest.json` |
| S205 | jiangsuiyang | `organ_T13_half_cl_20260908/short_checks_20260908/clip5_seed44_126/run/manifest.json` |
| S206 | jiangsuiyang | `organ_T13_half_cl_20260908/short_checks_20260908/frozen_bn_126/run/manifest.json` |
| S207 | jiangsuiyang | `organ_T13_half_cl_20260908/short_checks_20260908/frozen_bn_clip5_126/run/manifest.json` |
| S208 | jiangsuiyang | `organ_T13_half_cl_20260908/short_checks_20260908/lr003/run/manifest.json` |
| S209 | jiangsuiyang | `organ_T13_half_cl_20260908/short_checks_20260908/lr003_126/run/manifest.json` |
| S210 | jiangsuiyang | `organ_T13_half_cl_20260908/short_checks_20260908/lr003_clip5/run/manifest.json` |
| S211 | jiangsuiyang | `organ_T13_half_cl_20260908/short_checks_20260908/lr003_clip5_126/run/manifest.json` |
| S212 | jiangsuiyang | `organ_T13_half_cl_20260908/small_alpha_checks_20260908/alpha_0.01/run/manifest.json` |
| S213 | jiangsuiyang | `organ_T13_half_cl_20260908/small_alpha_checks_20260908/alpha_0.05/run/manifest.json` |
| S214 | jiangsuiyang | `organ_T13_half_cl_20260908/small_alpha_checks_20260908/alpha_0.1/run/manifest.json` |
| S215 | jiangsuiyang | `organ_T2_coverage_20260908/runs/cl_fg20/manifest.json` |
| S216 | jiangsuiyang | `organ_T2_coverage_20260908/runs/cl_fg40/manifest.json` |
| S217 | jiangsuiyang | `organ_T2_coverage_20260908/runs/stability_gate/manifest.json` |
| S218 | jiangsuiyang | `organ_T2_coverage_20260908/runs/sw_lr01_g0/manifest.json` |
| S219 | jiangsuiyang | `organ_T2_coverage_20260908/runs/sw_lr01_g01/manifest.json` |
| S220 | jiangsuiyang | `organ_T2_coverage_20260908/runs/sw_lr03_g0/manifest.json` |
| S221 | jiangsuiyang | `organ_T2_coverage_20260908/runs/sw_lr03_g01/manifest.json` |
| S222 | jiangsuiyang | `organ_T34_lr006_spatial_pair20_20260910/runs/spatial0/manifest.json` |
| S223 | jiangsuiyang | `organ_T34_lr006_spatial_pair20_20260910/runs/spatial001/manifest.json` |
| S224 | jiangsuiyang | `organ_batch8_probe_20260908/b8_spatial_on/run/manifest.json` |
| S225 | jiangsuiyang | `organ_metrics_half_20260909/prefix_T2/manifest.json` |
| S226 | jiangsuiyang | `organ_metrics_half_20260909/runs/ind_T4/manifest.json` |
| S227 | jiangsuiyang | `organ_t3_retention_probe_20260909/R0/manifest.json` |
| S228 | jiangsuiyang | `organ_t3_retention_probe_20260909/R1/manifest.json` |
| S229 | jiangsuiyang | `organ_t3_retention_probe_20260909/R2/manifest.json` |
| S230 | jiangsuiyang | `organ_t3_retention_probe_20260909/R3/manifest.json` |
| S231 | jiangsuiyang | `organ_t3_retention_probe_20260909/frozen/manifest.json` |
| S232 | jiangsuiyang | `replay_comparisons_20260911/runs/domain_zs-der/manifest.json` |
| S233 | jiangsuiyang | `replay_comparisons_20260911/runs/domain_zs-er/manifest.json` |
| S234 | jiangsuiyang | `replay_comparisons_20260911/runs/organ_zs-der/manifest.json` |
| S235 | jiangsuiyang | `tune_independent_A_07261_seed42_20260907_1303/independent/manifest.json` |
| S236 | jiangsuiyang | `tune_independent_A_07261_seed42_20260907_1303/oracle/manifest.json` |
| S237 | jiangsuiyang | `tune_independent_A_07261_seed42_20260907_1303/parity/independent/manifest.json` |
| S238 | jiangsuiyang | `tune_independent_A_07261_seed42_20260907_1303/parity/oracle/manifest.json` |
| S239 | jiangsuiyang | `tune_independent_A_07261_seed42_20260907_1303/parity_deterministic/independent/manifest.json` |
| S240 | jiangsuiyang | `tune_independent_A_07261_seed42_20260907_1303/parity_deterministic/oracle/manifest.json` |
| S241 | jiangsuiyang | `tune_independent_A_07261_seed42_20260907_1303/parity_pattern_f5_b10/independent/manifest.json` |
| S242 | jiangsuiyang | `tune_independent_A_07261_seed42_20260907_1303/parity_pattern_f5_b10/oracle/manifest.json` |
| S243 | jiangsuiyang | `tune_independent_A_07261_seed42_20260907_1303/parity_pattern_f5_b10/oracle_repeat/manifest.json` |
| S244 | jiangsuiyang | `tune_independent_A_07261_seed42_20260907_1303/parity_repeat/independent/manifest.json` |
| S245 | jiangsuiyang | `tune_independent_A_07261_seed42_20260907_1303/parity_repeat/oracle/manifest.json` |
| S246 | jiangsuiyang | `tune_independent_A_07261_seed42_20260907_1303/parity_repeat/oracle_repeat/manifest.json` |
| S247 | jiangsuiyang | `zs_independent_search_seed42_20260906_inclusive_aborted_20260906_2128/tuning/class_T2_c01/manifest.json` |
| S248 | jiangsuiyang | `zs_independent_search_seed42_20260906_inclusive_aborted_20260906_2128/tuning/class_T2_c02/manifest.json` |
| S249 | jiangsuiyang | `zs_independent_search_seed42_20260906_inclusive_aborted_20260906_2128/tuning/class_T2_c03/manifest.json` |
| S250 | jiangsuiyang | `zs_independent_search_seed42_20260906_inclusive_aborted_20260906_2128/tuning/class_T2_c04/manifest.json` |
| S251 | jiangsuiyang | `zs_independent_search_seed42_20260906_inclusive_aborted_20260906_2128/tuning/domain_C_c01/manifest.json` |
| S252 | jiangsuiyang | `zs_independent_search_seed42_20260906_inclusive_aborted_20260906_2128/tuning/domain_C_c02/manifest.json` |
| S253 | jiangsuiyang | `zs_independent_search_seed42_20260906_inclusive_aborted_20260906_2128/tuning/domain_C_c03/manifest.json` |
| S254 | jiangsuiyang | `zs_independent_search_seed42_20260906_inclusive_aborted_20260906_2128/tuning/domain_C_c04/manifest.json` |
| S255 | jiangsuiyang | `zs_independent_tuning_seed42_20260906/smoke_class_T2/manifest.json` |
| S256 | jiangsuiyang | `zs_independent_tuning_seed42_20260906/smoke_domain_B_skiptest/manifest.json` |
