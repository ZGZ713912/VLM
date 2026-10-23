# 全幅区域与时序 VLM：真实训练报告（2026-10-23）

本项目要从监控视频定位异常帧，并结合文本提供候选解释。旧路径使用整帧 CLIP 和运动
伪标签，官方 Avenue 上的真实检测能力偏低。本轮实现正常视频上的区域/时序学习，
自主获取 UCSD Ped2，完成 14 组实际训练和官方独立计分。新的完整模型显著改善了
Avenue 的历史基线；跨场景报警阈值和语言融合仍有明确不足。

## 数据、标注与实验隔离

| 数据集 | 训练视频/帧 | 测试视频/帧 | 测试异常帧 | 标注来源 |
|---|---:|---:|---:|---|
| Avenue | 16 / 15,328 | 21 / 15,324 | 3,712 | 官方 MATLAB `volLabel` 像素标注 |
| UCSD Ped2 | 16 / 2,550 | 12 / 2,010 | 1,648 | 官方 BMP 像素标注，与原始 `.m` 区间逐帧一致 |

Avenue 数据说明：[CUHK 发布页](https://www.cse.cuhk.edu.hk/leojia/projects/detectabnormal/dataset.html)。
Ped2 从 [UCSD 官方归档](http://www.svcl.ucsd.edu/projects/anomaly/UCSD_Anomaly_Dataset.tar.gz)
下载，只提取 Ped2。原归档 SHA256 为
`2329af326951f5097fdd114c50e853957d3e569493a49d22fc082a9fd791915b`。
转换工具检查全部 TIFF 帧和 BMP mask、二值性、帧号、分辨率，并对 FFV1 AVI 的每个
解码帧与原始 TIFF 做无损比对。mask 同时兼容 `001.bmp` 与 `frame001.bmp` 命名。

这里完成的是发布方人工标注的获取、转换与逐帧核验，没有将机器生成标签冒充新增人工
标注。正常训练集没有逐帧异常真值；Avenue 发布页还明确提到训练集可能含少量离群帧。
两套数据分别拟合各自模型，本轮不是跨数据集零样本迁移实验。

所有 seed 共用 split_seed=42，按完整训练视频固定分为 12 个拟合视频和 4 个留出视频；
Avenue 留出 06/07/08/10。模型按留出正常视频的预测误差选择 epoch，随后仅在这些
训练留出视频上校准分量尺度与 99% 分位阈值。模型、权重、prompt、融合系数和
14 组实验矩阵均在本轮首次查看官方测试成绩前固定。全部模型训练完成才开始统一测试。
测试标签没有参与拟合、校准或选优；所有消融都报告，没有按测试结果替换预先指定模型。

## 代码与训练路径

先设计模块与协议，见 [设计文档](region_temporal_design.md)。新代码按职责分布在
`datasets/`、`models/`、`prompts/`、`train/`、`eval/`、`utils/` 与入口脚本中。
原有训练、缓存、checkpoint 和 Web Studio 保留。

1. 全幅 RGB resize 到 CLIP 输入尺寸，不做中心裁剪，保留画面两侧。
   `ViT-B-32` 的局部 token 为 `(N,49,512)`，`ViT-B-16` 为 `(N,196,512)`。
   token 经视觉 projection 后归一化；局部 token 的检索语义仍需验证。
2. 同一网格提取 Farneback 光流的方向、幅值、分布与帧差等 18 维描述 `(N,R,18)`。
   它们是输入特征，不是异常真值。源视频、特征、预处理与编码器身份有独立校验。
3. 每个网格从拟合正常帧建立 512 条参考特征，k=3 查询局部外观与运动正常性。
4. 轻量模型融合区域视觉、标准化运动与外部文本；支持 concat/crossattn。
   单向 GRU 每个区域独立建模，用时刻 `<=t` 的信息预测 `t+1`。
5. 损失为 `appearance_prediction + 0.25 motion_prediction + 0.2 BCE + 0.05 contrastive`。
   外观预测用余弦距离，运动预测用 Smooth L1；BCE 区分正常特征与人为合成扰动，
   对比损失将正常视觉对齐正常 prompt。合成扰动不等价于真实异常行为监督。
6. 完整模型预先固定组合：外观 0.35、运动 0.35、时序预测 0.15、文本相似度差 0.15；
   合成分类头作为单独消融报告。每个分量取最高 3 个区域均值，5 帧因果平滑。

训练使用 AdamW、固定随机种子、确定性算子和独立 DataLoader generator；保存
model/optimizer/scheduler/Python/NumPy/Torch/CUDA/DataLoader RNG、配置、prompt、
数据划分和代码哈希。无 AMP 状态需要恢复，本轮训练为 FP32。
每组 12 epoch，共 **168 epoch、7,608 次 optimizer 更新**；不是单个模型训练 168 轮。

最终 `models/region_detector.py` 封装经过训练校准的完整评分。
输入区域外观 `(B,T,R,D)`、运动 `(B,T,R,18)`，输出必备字典字段
`anomaly_score(B), frame_score(B,T), embedding(B,D), explanation`，附带阈值 alert 和分量。
单次 forward 的每个 batch item 是一段完整视频；新视频会重置时序与平滑状态。
`RegionTemporalModel` 自身的分数是辅助分类头，部署完整方法应加载 `RegionDetector`。

## 官方结果

评估将所有测试视频帧的原始训练校准分数合并计算 Frame ROC-AUC/AP，不做逐测试
视频 min-max 归一化。这里只评帧检测，未计算官方空间定位指标。
两个数据集所有测试视频均含异常，Video AUC 未定义；不以 Video AP=1 宣称质量。
Ped2 的测试正例比例约 82%，AP 也不能当作分类准确率。

预先指定的主模型均为 scene / concat / ViT-B-32 / seed=42：

| 模型 | Avenue AUC | Avenue AP | Ped2 AUC | Ped2 AP |
|---|---:|---:|---:|---:|
| 完整模型（预先指定） | **0.8215** | **0.6921** | **0.8907** | **0.9736** |
| 局部外观 | 0.7340 | 0.4726 | 0.8646 | 0.9654 |
| 局部运动 | 0.8645 | 0.7083 | 0.8257 | 0.9594 |
| 学习后的时序预测 | 0.8240 | 0.6543 | 0.8553 | 0.9657 |
| 冻结文本相似度差 | 0.6188 | 0.4486 | 0.9159 | 0.9811 |
| 合成扰动分类头 | 0.6924 | 0.5288 | 0.8304 | 0.9596 |
| 外观 + 运动 | 0.8228 | 0.6694 | 0.8720 | 0.9695 |

Avenue 的已核验历史 scene 零样本基线为 AUC/AP=0.6263/0.3799，正常全帧特征库
为 0.5744/0.2923。本轮主模型分别增加 AUC **0.1951 / 0.2471**。
按相同视频成对 bootstrap 2,000 次，AUC 增量 95% 区间分别为
`[0.0475,0.3309]`、`[0.1637,0.3150]`。这些是当前基准的探索性区间，不代表新场景性能。

scene/concat 的两 seed（42/43）AUC 均值±样本标准差：Avenue **0.8171±0.0062**，
Ped2 **0.8863±0.0063**，仅两个 seed。crossattn 对应 **0.8170±0.0064 / 0.8866±0.0061**，
目前没有明显优于 concat。label、contrast 消融和全部分数保存在运行汇总中。
第二骨干 ViT-B-16 在 Ped2 上完整模型 AUC=0.8278，未优于 ViT-B-32；没有只报告获胜模型。

![训练曲线与固定消融](../results/region_v1_20261023/learning_and_ablation.png)

## 当前限制与失败证据

融合未稳定超过单分量：Avenue 运动分量的 AUC 高于完整模型，Ped2 文本分量也高于
完整模型。因此不能宣称语言或时序加入后一定带来净收益，模型改进主要来自更合适的
区域/运动正常性表征。全模型相对历史基线的改进并不能隔离每一项机制的因果贡献。

仅正常训练留出视频校准的 99% 阈值，在测试场景中存在明显偏移：

| 主模型 | 正常帧误报率 FPR | 异常帧召回 TPR |
|---|---:|---:|
| Avenue | 0.67% | 26.27% |
| Ped2 | 45.30% | 93.63% |

Avenue 18 的 294 帧在官方标注中全异常，但当前阈值零报警。完整失败视频与无标注推理
产物一并保留，不能只用 AUC 或成功样本宣称已经适合实际报警。
文本解释为检索候选，没有经过人工事件语义核验；网格也没有空间定位质量保证。
下一轮应使用新的独立验证材料研究场景校准与可学习融合，不据本轮测试表反复挑权重。

## 产物

- 主模型：`checkpoints/region_v1_avenue_scene_concat_s42/detector.pt`、
  `checkpoints/region_v1_ped2_scene_concat_s42/detector.pt`。
- 完整矩阵：`results/region_v1_20261023/summary.json`、`summary.md`；
  主模型对比：`primary_results.md`；seed 与配对区间：`analysis.json`。
- 每组：config、split_manifest、环境、source_hashes、history、训练收据、正常校准分量，
  `official/` 内有完整帧标签、7 组分数、分量、逐视频指标、置信区间、解释与时间轴。
- 原始 Ped2、转换记录和标注 manifest：`data/downloads/`、`data/UCSDped2/`、
  `data/labels_ped2_official/testing/manifest.json`。
- `results/region_v1_20261023/reproduction/`：实际训练源码快照、文件哈希、pip freeze、
  容器镜像 ID 和两个公开骨干的 revision/权重 SHA256。
- 三段完整演示：`results/region_v1_demo/avenue01.mp4`、`ped2_Test001.mp4`、
  `avenue18_missed_anomaly.mp4`。附源/输出哈希、完整帧数收据及阈值。
- `results/region_v1_inference_avenue18/`：不访问标签的独立视频推理，与正式评估
  逐帧分数最大误差 **0.0**。
- 最终完整验证 **95 passed**，包括真实数据/GPU 集成，见
  `results/verification/region_final_20261023.xml`。覆盖两种融合的因果性、随机恢复、
  梯度和真实更新、视频隔离、哈希/骨干/标签错误拒绝及完整导出评分接口。

## 复现

本机命令在 `vlm_ws-vlm-dev-1` 容器内、目录 `/workspace/vlm_ws` 执行。
环境 Python 3.10、PyTorch 2.3.1/CUDA 12.1、OpenCLIP 2.26.1、RTX 4060 Ti。
已缓存权重可设置 `HF_HUB_OFFLINE=1`；首次获取权重请不设置离线模式。
本轮还通过 Context7 的 MCP HTTP 服务查询了 PyTorch/OpenCLIP 接口，并核对安装源码。

```bash
# 已有 preparation.json 和官方 manifest 时核验归档、视频和标签，复用且不覆盖
python tools/prepare_ped2.py

# full-frame region caches：对每个数据集训练/测试各运行一次；不读取标签
python tools/extract_region_features.py --video-dir data/Avenue_Dataset/training_videos --out data/features_region_v1/avenue/vitb32/training
python tools/extract_region_features.py --video-dir data/Avenue_Dataset/testing_videos --out data/features_region_v1/avenue/vitb32/testing
python tools/extract_region_features.py --video-dir data/UCSDped2/training_videos --out data/features_region_v1/ped2/vitb32/training
python tools/extract_region_features.py --video-dir data/UCSDped2/testing_videos --out data/features_region_v1/ped2/vitb32/testing
# 第二骨干：对应两条 Ped2 命令改 out 的 vitb32 为 vitb16，并增加
# --model ViT-B-16 --pretrained laion2b_s34b_b88k

# 新单组训练，保护现有结果；train 默认不打开测试集
python scripts/train_region_model.py --config configs/region_temporal.yaml --run-name my_region_run
python scripts/train_region_model.py --config configs/region_temporal.yaml --run-name my_region_run --evaluate-only
# 如需先停止在某 epoch 后再恢复，保持配置完全一致
python scripts/train_region_model.py --config configs/region_temporal.yaml --run-name my_resume_run --stop-after-epoch 2
python scripts/train_region_model.py --config configs/region_temporal.yaml --run-name my_resume_run --resume

# 完整预声明矩阵：重跑请复制 campaign 配置并更改所有 run 名称
python scripts/run_region_experiments.py --dry-run
OMP_NUM_THREADS=2 MKL_THREADING_LAYER=GNU python scripts/run_region_experiments.py --phase train
OMP_NUM_THREADS=2 MKL_THREADING_LAYER=GNU python scripts/run_region_experiments.py --phase evaluate
python scripts/report_region_campaign.py

# 无标注推理与演示
python scripts/predict_region_video.py --checkpoint checkpoints/region_v1_avenue_scene_concat_s42/detector.pt --video path/to/video.avi --out-dir results/new_prediction
python scripts/visualize_region_result.py --run results/region_v1_avenue_scene_concat_s42 --video-id 01 --output results/new_demo/avenue01.mp4

VLM_REAL_INTEGRATION=1 HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 MKL_THREADING_LAYER=GNU python -m pytest -q
```

train/evaluate-only 均校验配置、prompt、数据 manifest 与代码身份；resume 保存精确随机
状态。评估目录存在会拒绝覆盖；campaign 重入只复用完整产物，失败返回非零并保留日志。
源代码快照用于重现实验的原始版本，后续修改算法须使用新的 run 和缓存版本。
