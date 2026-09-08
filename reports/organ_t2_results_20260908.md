# Organ-CL T2 标注率与首轮遗忘实验结果

四组单轮 T3 遗忘实验于 2026-09-08 13:27（Asia/Shanghai）全部完成并停止。没有启动新的 80 轮持续学习正式训练。两组先前启动的 T2 independent 80 轮训练也已完成。

## 1. T2 在 T3 第一轮后的遗忘

以下均为按病例平均的 3D **前景验证 Dice**，不包含背景 Dice，不使用测试集选参。T2 前景标注覆盖率统一为 19.6609%。

| 初始 LR | Global | T2 学完自身任务 | T3 训练一轮后的 T2 | 变化：后减前 | 保留率 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.01 | 0 | 0.1747 | 0.0519 | -0.1228 | 29.70% |
| 0.03 | 0 | 0.3005 | 0.2021 | -0.0984 | 67.27% |
| 0.01 | 0.1 | 0.0745 | 0.1424 | +0.0679 | 191.25% |
| 0.03 | 0.1 | 0.3979 | 0.0000 | -0.3979 | 约 0% |

最后一行未四舍五入的 T2 Dice 为 `3.332019979504167e-10`。

其他已见任务的终点评估用于补充解释：

| LR / Global | T3 一轮后 T1 | T3 一轮后 T3 |
| --- | ---: | ---: |
| 0.01 / 0 | 0.3327 | 约 0 |
| 0.03 / 0 | 0.3112 | 0.3452 |
| 0.01 / 0.1 | 0.4403 | 0.6474 |
| 0.03 / 0.1 | 0.6690 | 0.4037 |

**解释：**LR=0.03、Global=0.1 的 T2 初始学习最好，却在 T3 一轮后基本完全遗忘；不能用初始 Dice 选择持续学习配置。LR=0.03、Global=0 的 T2 终点 Dice 最高，但仍相对下降 32.73%。LR=0.01、Global=0.1 的保留率大于 100%，主要因为初始 T2 只有 0.0745；终点 0.1424 仍很低，不能据此宣称保留问题解决。Global 与 LR 存在配置相关的表现差异，不能从这四组单种子结果推出 Global 普遍有害。没有一组同时达到 T2 初始 Dice 至少 0.30 且首轮保留率至少 80%。

## 2. 独立训练：前景标注率从约 20% 增至 40%

两组均使用 T2 从零训练、80 epochs、LR=0.003、Global=0.1、clip=1；按验证集选择 checkpoint，测试集仅用于事后报告。训练配方与上面的 LR sweep 不同，不能横向混作相同设置。

| T2 前景覆盖率 | 全像素标注率 | 最佳验证 Dice | 对应测试 Dice | 最佳 epoch（一基） |
| ---: | ---: | ---: | ---: | ---: |
| 19.6609% | 10.5194% | 0.0710 | 0.0496 | 74 |
| 40.4353% | 10.8000% | 0.0690 | 0.0488 | 65 |

扩充只新增 30,523 个训练前景像素；原有前景和 1,115,520 个背景标注像素不变。40% 是逐前景切片至少覆盖 40% 的生成目标，实际总前景覆盖率为 40.4353%；这不是人工标注时间的测量。

**结论范围：**增加标注在这套表现较差的配方下没有带来可见改善。两组均只有约 0.05 测试 Dice，且没有多种子重复，因此不能推出增加标注总体无效。当前更迫切的问题是验证集上的泛化表现和跨任务保留，而不是直接投入更多标注或长程训练。

## 3. 实验协议及验证

- 四组共享：seed=42，T1/T2 各 20 epochs，batch=4，SGD momentum=0.9、weight decay=0.0001、poly exponent=0.9，gradient norm clip=1，PCE=1、Spatial/GD=0，DER++ alpha/beta=0.5/0.5，buffer=128，replay minibatch=4。每轮验证，按当前任务验证 Dice 选择并成对恢复模型和回放状态。
- 用户将 T3 从 20 轮改为 1 轮时，一组已完成第二轮，首轮 checkpoint 被覆盖。因此所有候选统一从已保存的 T2 模型/回放状态新开 T3 分支，transition seed=44；没有重跑 T1/T2。旧 checkpoint 未保存 RNG，这不是原始不间断训练首轮的精确重放。
- T3 保留 20 轮的学习率衰减跨度，只执行第一轮，不把学习率在一轮内衰减至零。四个新分支均核实为 stage=2、epoch=0、356 次更新，T2 前后指标与持久化验证矩阵一致。
- 数值/标注测试先前通过 32 项；Global=0 真正跳过 replay global 计算的修复通过 3 项相关测试。真实 GCO、DER++ 状态与 BN smoke、两阶段端到端 smoke 已通过。单轮分支和两个 independent 的完整 epoch 数由导出程序再次检查通过。
- 原始 replay batch=8 短检曾 OOM；改为 4 后短检通过。最后一组曾因固定 GPU 5 显存等待延迟，之后转到 GPU 7 完成。这些是运行事件，不是方法表现。全部旧数据、checkpoint、失败日志和停止记录保留在服务器。
- 自动正式训练已关闭。单轮结果只能刻画早期遗忘，不能代表完整 T3/T4 训练后的持续学习成绩。

## 4. 复现与公开范围

完整精度汇总：`results/organ_t2_epoch1_20260908/summary.json`；另有 `forgetting.csv` 与 `independent.csv`。`export_t2_results.py` 仅导出任务级标量和矩阵，不导出病例级明细。

在已有数据和环境中，先生成两档训练标注：

```bash
python prepare_t2_coverage.py --data-root "$DATA_ROOT" --sparse-root "$SPARSE_ROOT" --output "$ANNOTATIONS"
```

每个候选先通过共享 runner 训练 T1/T2：

```bash
python main.py --setting-run --data-root "$DATA_ROOT" --sparse-root "$ANNOTATIONS/fg20" \
  --output "$T12_OUTPUT" --device cuda:0 --method zs-derpp --max-task 2 \
  --epochs-per-task 20 --batch-size 4 --workers 4 --seed 42 --lr "$LR" \
  --grad-clip-norm 1 --zs-global-weight "$GLOBAL" --zs-spatial-loss-weight 0 \
  --der-alpha .5 --der-beta .5 --der-buffer-size 128 --der-minibatch-size 4 --validate-each-epoch
```

保持上述设置，将 `--output` 改为新路径、`--max-task` 改为 3，并增加 `--t3-one-epoch-from "$T12_OUTPUT"`，即可只运行标准化的一轮 T3 并生成验证矩阵。LR/Global 取表中的四个组合。Independent 使用 `--method zs-sequential --organ-task T2 --epochs-per-task 80 --lr .003 --zs-global-weight .1 --test-evaluation`，不传 `--max-task` 和 DER++ 控制，分别使用两档标注目录，其余共同设置不变。

公开范围为训练代码、生成/执行/导出脚本、协议、标量指标与矩阵。医学图像、标注数组、模型、回放 buffer、私有故障快照和原始运行日志不公开。发布脚本使用参数化数据路径；病例级结果未包含在本报告或导出中。
