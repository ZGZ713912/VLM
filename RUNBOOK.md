# VLM-VAD RUNBOOK

本项目的最小可复现操作手册：从零样本验证到训练、评估、可视化。
所有命令均在**仓库根目录** `/workspace/vlm_ws` 执行。

- 配置：`configs/experiment.yaml`（训练/数据/模型/评估），`configs/prompts.yaml`（prompt 词表，加载时自动合并进 `cfg.prompt`）
- 数据：`data/Avenue_Dataset/`（`training_videos` 全正常；`testing_videos` + `ground_truth/testing_label_mask` 异常标注）
- 依赖：首次运行 CLIP 相关脚本会下载 open_clip `ViT-B/32`（`laion2b_s34b_b79k`，约 350MB）；建议 GPU

---

## 0. 环境自检

```bash
cd /workspace/vlm_ws
python -c "import cv2, decord, scipy, omegaconf, torch, open_clip; print('deps ok')"
python -m pytest tests/test_smoke.py -q      # 应 10 passed（CPU 秒级，不下载权重）
```

冒烟测试用 `FakeCLIPBackbone`，只验证配置/prompt 极性/指标/forward 形状/loss 反向，
**不产生任何科研指标**。

### 0.1 离线运行（无外网）

CLIP 权重已缓存在 `$HF_HOME/hub`（本机为 `/tmp/devhome/.cache/huggingface/hub`）。
但 open_clip 默认仍会联网检查更新，无网络时会报
`HTTPSConnectionPool ... Connection refused`。开启离线模式即可用缓存：

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

### 0.2 CPU 运行

`configs/experiment.yaml` 的 `device` 用 `"auto"`（有 GPU 用 GPU，否则自动 CPU）。
无 GPU 时所有 CLIP 脚本仍可用，只是编码慢；建议先做第 2 节的特征缓存再跑多个实验。


---

## 1. 数据路径（Design A）

`configs/experiment.yaml` 的 `data:` 段按 split 显式声明路径：

```yaml
data:
  train_split: "training"
  eval_split: "testing"
  train:
    video_path: "data/Avenue_Dataset/training_videos"
    mask_path: null                 # 训练视频全正常 → 帧标签全 0
  test:
    video_path: "data/Avenue_Dataset/testing_videos"
    mask_path: "data/Avenue_Dataset/ground_truth/testing_label_mask"
```

`build_split_dataloader(cfg.data, split)` 会把 split 归一化为 `training`/`testing`，
再映射到 `train`/`test` 段读取 `video_path`/`mask_path`。
mask 命名约定：`{int(video_id)}_label.mat`（`volLabel` 变量）。

---

## 2. 可选加速：预提取 CLIP 特征（推荐）

提取后 `build_split_dataloader` 自动切换到 `FeatureDataset`（约 10× 加速）。

```bash
# testing split（评估 / zero-shot 用）
python tools/extract_video_features.py \
  --video-dir data/Avenue_Dataset/testing_videos \
  --mask-dir  data/Avenue_Dataset/ground_truth/testing_label_mask \
  --split testing --out data/features/testing --device cuda

# training split（训练用；无 mask）
python tools/extract_video_features.py \
  --video-dir data/Avenue_Dataset/training_videos \
  --split training --out data/features/training --device cuda
```

产物：`data/features/{split}/{video_id}.pt`，每个 shape `(N, D)`。
可选：预编码文本 prompt（复用）：

```bash
python tools/extract_text_features.py \
  --config configs/prompts.yaml \
  --out data/features/text/prompts.pt --device cuda
```

> 若跳过本节，`use_feature: false` 时走 `VideoDataset`（每轮实时解码视频），
> 冻结 backbone 时较慢，但结果等价。

---

## 3. 阶段 A：Zero-shot prompt 策略验证

目标：得到不同 prompt 策略下的零样本表现 + 可视化 + 结论。

### A1. 单类型 zero-shot

```bash
python scripts/zero_shot_test.py --config configs/experiment.yaml --prompts label
python scripts/zero_shot_test.py --config configs/experiment.yaml --prompts scene
python scripts/zero_shot_test.py --config configs/experiment.yaml --prompts contrast
python scripts/zero_shot_test.py --config configs/experiment.yaml --prompts all
```

### A2. 一次性消融（推荐，直接出结论表）

```bash
python scripts/prompt_comparison.py --config configs/experiment.yaml --save-plots
```

实验定义来自 `configs/prompts.yaml` 的 `experiments` 段
（`label_only` / `scene_only` / `contrast_only` / `label_scene` / `label_contrast` /
`scene_contrast` / `all_types`），改 yaml 即可增删。

### 产物

| 路径 | 内容 |
|---|---|
| `results/zero_shot_{prompts}/summary.json` | 单次 zero-shot 的 metrics + 耗时 |
| `results/zero_shot_{prompts}/config.yaml` | 配置快照 |
| `results/zero_shot_{prompts}/zero_shot_metrics.json` | 指标 |
| `results/zero_shot_{prompts}/zero_shot_explanations.json` | 每个视频的解释 |
| `results/zero_shot_{prompts}/zero_shot_plots/{vid}_heatmap.png` | 逐帧异常曲线 + GT 色带 |
| `results/zero_shot_{prompts}/zero_shot_plots/{vid}_prompts.png` | 每个 prompt 的相似度 |
| `results/prompt_comparison.md` | 各组对比表 + Best by Frame AUC |
| `results/prompt_comparison.json` | 机器可读结果 |
| `results/prompt_comparison/{exp}/` | 每个实验的子目录（metrics/explanations/plots） |

### 结论书写要点

- 比较各组的 `frame_auc/frame_ap`（帧定位）与 `video_auc/video_ap`（视频判别）。
- prompt 数量≠性能：`all_types` 常因语义噪声低于 `scene_only`。
- zero-shot 帧级 AUC 可能 < 0.5（语义 prompt 与像素异常不完全对齐），
  这是正常现象，用作"为什么需要微调"的论据。

---

## 4. 阶段 B：训练模型

### B1. Prompt 处理

只改 `configs/prompts.yaml`（`templates` / `expansions` / `hard_negatives` /
`normal_keywords` / `abnormal_keywords`），**不改代码**。模型与 loss 均从同一份
`cfg.prompt` 取数，保证前后一致。

### B2. 关键超参（`configs/experiment.yaml`）

```yaml
train:
  epochs: 10
  lr: 1.0e-4
  lr_backbone: 1.0e-5
  weight_decay: 1.0e-4
  warmup_steps: 50
  grad_clip: 1.0
  amp: false
  loss:
    w_bce: 1.0
    w_contrastive: 0.5
    skip_bce_when_no_pos: true
model:
  backbone: {freeze_vision: true, freeze_text: true}
  fusion:   {type: "concat"}       # concat | gated | crossattn
  temporal: {type: "identity"}     # identity | transformer
```

### B3. 训练

```bash
python train.py --config configs/experiment.yaml --run-name exp01

# 续训（从 last.pt 之后继续）
python train.py --config configs/experiment.yaml --resume checkpoints/exp01/last.pt
```

训练流程：`forward → loss → backward → optimizer.step` 每步；
每 epoch 做验证（frame/clip AUC&AP），按 **val `clip_auc`** 保存 `best.pt`/`last.pt`；
训练结束后自动用 `best.pt` 在测试集评估。

### B4. 结果验证

```bash
# 单独用 best.pt 干净评估
python eval.py --config configs/experiment.yaml --ckpt checkpoints/exp01/best.pt

# 不带 --ckpt：下游随机初始化，仅 zs_* 零样本保底有效
python eval.py --config configs/experiment.yaml
```

### 产物

| 路径 | 内容 |
|---|---|
| `logs/exp01/` | TensorBoard（`train`/`val` 曲线） |
| `checkpoints/exp01/{best,last}.pt` | 模型/优化器/调度器/matcher/prompt 状态 |
| `results/exp01/config.yaml` | 合并后的完整配置快照（可复现） |
| `results/exp01/metrics.json` | frame/clip/video AUC&AP + `zs_*` 保底 |
| `results/exp01/explanations.json` | 每个视频的异常原因 |
| `results/exp01/plots/{vid}_{heatmap,prompts}.png` | 可视化 |

检查点：
1. `metrics.json` 的 trained 指标应 **≥ `zs_*` 保底**，否则训练有问题。
2. 对比 "zero-shot vs fine-tuned"。
3. 消融：改 `fusion.type` / `temporal.type` 各训一次，填对比表。

---

## 5. 可视化演示（可解释性）

需要训练产出的 checkpoint：

```bash
python scripts/visualize_demo.py --config configs/experiment.yaml \
  --ckpt checkpoints/exp01/best.pt --videos 01 06 18
```

产物：`results/demo/{vid}_demo.mp4`、`{vid}_demo.gif`、`{vid}_top_frames.png`、
`timeline_{vid}.png`、`index.md`（可直接贴 PPT）。

---

## 6. 结论书写框架

1. 实验设置：数据集/标签来源/backbone/特征路径/硬件，固定 seed。
2. Zero-shot 表（各 prompt 策略）。
3. Fine-tuned 表（frame/clip/video AUC&AP）+ `zs_*` 对照。
4. 消融：prompt 类型、fusion（concat/gated/crossattn）、temporal（identity/transformer）。
5. 可解释性样例（`explanations.json` + 热力图）。
6. 局限与后续。

### 已知局限（需在结论中说明）

1. **验证集=测试集**：`train.py` 用 `eval_split: testing` 作验证并据此选 `best.pt`，
   存在"用测试集选模型"的偏差；严格协议应另分验证集。
2. **像素级监督未启用**：`pixel_mask (T,H,W)` 在 datasets 中已产出但无任何消费方，
   当前只有帧级/片段级监督，无像素级定位。
3. **训练视频全正常**：training 无异常标注，训练依赖对比损失与 `skip_bce_when_no_pos`。

---

## 7. 命令速查

```bash
# 冒烟
python -m pytest tests/test_smoke.py -q

# 预提取特征（可选）
python tools/extract_video_features.py --video-dir data/Avenue_Dataset/testing_videos \
  --mask-dir data/Avenue_Dataset/ground_truth/testing_label_mask \
  --split testing --out data/features/testing --device cuda

# Zero-shot + 消融
python scripts/zero_shot_test.py --config configs/experiment.yaml --prompts all
python scripts/prompt_comparison.py --config configs/experiment.yaml --save-plots

# 训练 / 评估 / 演示
python train.py --config configs/experiment.yaml --run-name exp01
python eval.py  --config configs/experiment.yaml --ckpt checkpoints/exp01/best.pt
python scripts/visualize_demo.py --config configs/experiment.yaml \
  --ckpt checkpoints/exp01/best.pt --videos 01 06 18
```
