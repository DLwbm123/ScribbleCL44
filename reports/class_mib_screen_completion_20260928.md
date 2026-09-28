# Class-CL ER / feature-DER + MiB：短程配对验证

2026-09-28 已完成。四组 T2 五轮实验均退出码 0，训练轮次完整、损失有限，
两个 MiB 真实数据启动检查通过。最后一组于北京时间 17:55:01 结束。
整个有限队列约 25 分 52 秒；未启动完整训练或其他方法实验。

## 验证结果

以下均为前景 validation Dice，不含背景；不是测试集结果。

| 方法 | T1 | T2 | 两任务平均 | 相对配对对照的平均增益 |
|---|---:|---:|---:|---:|
| ZS-ER | 0.000000 | 0.357665 | 0.178833 | — |
| ZS-ER + MiB | 0.746046 | 0.460962 | 0.603504 | +0.424671 |
| ZS-DER（guarded feature-DER） | 0.000000 | 0.280793 | 0.140396 | — |
| ZS-DER（guarded feature-DER）+ MiB | 0.576615 | 0.294645 | 0.435630 | +0.295234 |

两种方法的短程旧任务保持均明显改善，新任务 T2 也高于各自对照。
ER 的初始 T1 验证值为 0.719282，DER 为 0.570008；两种方法使用各自的
历史 T1 起点，因此本实验支持各方法内部的有/无 MiB 配对比较。
T1 高于初始值只是本次观测，不作普遍正迁移或统计显著性结论。

## 协议及证据边界

- 各组复用对应方法已完成的 80 轮 T1 权重及阶段末 reservoir；MiB 在 T1
  不启用，因此 T1 前缀可复用。任务边界 RNG 重置为 seed 42，非精确轨迹续跑。
- 每种方法先做两个真实数据更新的 MiB 检查，再独立从同一 T1 起点执行
  无 MiB / 有 MiB 两组 T2 五轮训练。检查权重不用于后续训练。
- 固定 seed 42、batch 4、workers 0、SGD LR .03、PCE/Global/Spatial=1/.1/0、
  buffer 64、replay minibatch 4、MiB KD 权重 10，没有调参选择。
- ER 保持 replay PCE=1、feature=0；DER 保持 feature MSE=.5、replay PCE=0，
  并沿用原数值保护与梯度裁剪 5。没有混入此前 BN/旧头冻结或 feature 刷新改进。
- MiB 复用项目既有的稀疏无偏分类与教师蒸馏实现；不是完整原论文复现的新声明。
  `--with-mib` 关闭时原方法行为不变，既有 DER++ 方法保持不变。
- 按原新任务 T2 前景验证值选择 checkpoint；ER 两组和 DER+MiB 选择第 5 轮，
  DER 对照选择第 4 轮。报告所选 checkpoint 对 T1/T2 的验证结果。
- 单种子、单一权重、只到 T2 的短程验证，不能替代完整三任务结果，也不能填为
  正式测试成绩。没有自动扩展实验、重试、完整训练或循环监测。
- 已知 CUDA grid-sample backward 的非确定性警告仍存在。

可复现协议见 `class_mib_screen_runtime/README.md`，有限队列及执行断言见
`class_mib_screen_runtime/job.py`，共享训练开关见
`class_replay_runtime/runner_core.py`，四组去隐私配置、逐轮标量损失与验证结果见
`results/class_mib_screen_20260928/aggregate.json`。
公开内容不包含原始数据、患者级指标、checkpoint、回放样本、私有路径或原始日志。
