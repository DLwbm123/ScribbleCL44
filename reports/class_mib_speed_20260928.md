# Class-CL MiB：A100 与 RTX PRO 5000 72GB Blackwell 实测

2026-09-28 完成必要文件迁移及四次短程测速。原 A100 正式训练保持运行，
没有迁移其活动进程、停止原训练或启动新服务器上的正式实验。

## 相同配置的训练循环速度

| 方法 | A100 40GB 秒/批 | RTX PRO 5000 72GB 秒/批 | 吞吐提升 | 同训练量耗时减少 |
|---|---:|---:|---:|---:|
| ZS-ER + MiB | 0.376611 | 0.322740 | 16.69% | 14.30% |
| ZS-DER + MiB | 0.416268 | 0.364291 | 14.27% | 12.49% |

新服务器在当前任务和配置下确有加速，约为 1.14–1.17 倍吞吐。
包含模型/数据载入、30 次更新和阶段验证的单次运行耗时分别为：
ER 22.58 → 18.42 秒；DER 33.97 → 30.47 秒。
这不是完整 50-epoch 实验计时，不能直接当作其实际完成时间。

## 测速协议

两台机器各在 GPU0 顺序运行 ER 和 guarded feature-DER。每次恢复同一方法的
历史 T1 模型及 reservoir，执行 30 次真实 T2 更新。batch=4，workers=0，
CPU threads=4，FP32，seed=42，H5 cache，LR=.03，PCE/Global/Spatial=1/.1/0，
MiB KD=10，buffer=64，replay minibatch=4；ER alpha/beta=0/1，DER=.5/0，
DER 使用既有数值保护与梯度裁剪 5。未调整 batch、精度、数据、目标函数或模型。

在相邻 `optimizer.zero_grad` 边界同步 CUDA 并计时，覆盖数据准备、前向、
损失、反向、优化器及回放更新。30 次更新产生 29 个间隔，排除最初 5 个，
报告余下 24 个间隔均值。中途不做验证，最终验证不计入每批耗时。
脚本断言真实更新数为 30、MiB KD 为正，各次测速均成功退出。
测速输出仅用于工程速度比较，不用于方法性能或 checkpoint 选择。

环境差异：A100 使用 PyTorch 2.6.0+cu124，新服务器使用支持 Blackwell 的
PyTorch 2.7.1+cu128。因此这是两套服务器运行环境的实测对比，不能将全部差值
仅归因于 GPU。A100 的另两张卡上原正式训练持续运行，CPU/存储可能共享；
新服务器也有其他 GPU 作业。每方法每机器只测一次，未给统计置信区间。
CUDA grid-sample backward 的已知非确定性警告仍存在。

## 必要迁移范围与环境

最终保留 44 个源文件及数据文件，总计 8,846,318,823 字节，约 8.24 GiB：
四份 Class/whole-heart H5、三份稀疏标注、ER/DER 各自 T1 恢复所需文件，
以及训练循环直接依赖的 Python 源码。没有搬其他场景数据、历史实验集合、
完整仓库或预训练模型。T1 恢复文件用于真实历史回放与 MiB 教师测速。

使用 SSH 加密的 rsync 直接传输，退出码 0；不删除源文件，不做全量哈希。
中转过的部分数据由 rsync 复用，直接传输阶段约 107 秒。
仅本次新复制的冗余检查/旧启动脚本已移除，原服务器文件不变。
目标位于用户指定的数据盘，实际 ext4 挂载和写入/读取探针通过。
私钥未复制或写入服务器；临时 SSH agent 在传输结束后关闭。

复用新服务器现有 PyTorch/CUDA，通过独立轻量环境补充 h5py、setproctitle、
gco-wrapper；现有环境未修改。缓存、临时文件、环境和所有输出均留在指定数据盘。

## 复现与证据

复用 `class_replay_runtime/runner_core.py` 及 ER/guarded-DER 依赖，运行
`class_mib_screen_runtime/speed_job.py`，由其调用 `benchmark.py`。
私有 `plan.json` 提供 data、sparse、resume、source_er、source_der 路径；
control 目录含上述脚本和名为 main 的 Python 入口，外层传入依赖 PYTHONPATH。
所有训练参数通过环境传递，训练进程名称为 run/job。

公开原始计时间隔及聚合结果：`results/class_mib_speed_20260928/my-gpu.json`
和 `pro5000.json`。不公开患者数据、患者级验证值、checkpoint、回放内容、
私有配置、凭据或完整训练日志。
