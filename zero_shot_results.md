# Zero-Shot 视频异常检测结果

> `results/*` 被 `.gitignore` 忽略，原始 JSON/图不入库；本文件记录全部数值结果。
>
> - 数据：CUHK Avenue `testing`（21 视频 / 15,324 帧 / 3,712 异常帧 / 968 clips）
> - Backbone：OpenCLIP `ViT-B-32` / `laion2b_s34b_b79k`（冻结）
> - 代码版本：`ce7fb2e`；脚本：`scripts/zero_shot_test.py` → `eval/zero_shot.py`
> - 配置：`configs/experiment.yaml` + `configs/prompts.yaml`

---

## 1. 实验设置

### 1.1 数据与基线

| 项 | 值 |
|---|---|
| 测试视频 / 帧 | 21 / 15,324 |
| 异常帧 | 3,712（24.22%） |
| clips / 异常 clips | 968 / 289（29.86%） |
| 全异常视频 | `08`（36/36）、`18`（294/294） |

| 级别 | 样本单位 | 分数 | 随机基线 |
|---|---|---|---|
| frame | 每帧 | 重叠 clip 按真实帧下标平均 | AUC 0.5 / AP 0.2422 |
| clip | 每 clip | `max(frame_scores)` | AUC 0.5 / AP 0.2986 |
| video | 每视频 | 所有 clip 的最大分 | 单类，**未定义** |

### 1.2 Prompt 组

| 组 | 模板 | K | 异常(+1) | 正常(−1) | 示例 |
|---|---|---:|---:|---:|---|
| label | `a person {label}` | 14 | 8 | 6 | `a person fighting` / `a person walking` |
| scene | `a {scene} scene` | 9 | 4 | 5 | `a suspicious crowd scene` / `a normal street scene` |
| contrast | `a video of {activity}` | 8 | 4 | 4 | `a video of abnormal behavior` / `a video of normal behavior` |
| all | 三者并集 | 31 | 16 | 15 | — |

### 1.3 打分

```
sims (B,T,K) = CLIP.encode_video(video) · Pool(CLIP.encode_text(prompts))ᵀ
s_t          = mean_{k∈Abnormal} sims[t,k] − mean_{k∈Normal} sims[t,k]
```

- 极性由关键词规则确定：`abnormal_keywords → +1`，`normal_keywords → −1`
  （`train/losses.py:31-67`）；本次所有 prompt 均分到 ±1。
- 池化：`mean` = 77 个 token 平均；`eos` = CLIP 官方 EOS 表示
  （`eval/zero_shot.py:157-170`，`models/backbone.py:95`）。
- 相似度约 0.20–0.30，`s_t` 约 0.005–0.03；指标只看排序。

---

## 2. 主结果

| Prompt 组 | 池化 | K(+/-) | frame AUC | frame AP | clip AUC | clip AP | video AUC | video AP |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| label | mean | 14 (8/6) | 0.4535 | 0.2406 | 0.5019 | 0.3278 | 0.0 | 0.0 |
| label | **eos** | 14 (8/6) | 0.4793 | 0.2526 | 0.5247 | 0.3507 | 0.0 | 0.0 |
| scene | mean | 9 (4/5) | 0.4953 | 0.2768 | 0.5765 | 0.4041 | 0.0 | 0.0 |
| scene | **eos** | 9 (4/5) | **0.6494** | **0.3721** | **0.6906** | **0.4879** | 0.0 | 0.0 |
| contrast | mean | 8 (4/4) | 0.3017 | 0.1780 | 0.3616 | 0.2545 | 0.0 | 0.0 |
| contrast | eos | 8 (4/4) | 0.3734 | 0.1970 | 0.3999 | 0.2585 | 0.0 | 0.0 |
| all | mean | 31 (16/15) | 0.4534 | 0.2457 | 0.5099 | 0.3395 | 0.0 | 0.0 |
| all | eos | 31 (16/15) | 0.5369 | 0.2969 | 0.5934 | 0.4069 | 0.0 | 0.0 |
| *随机基线* | — | — | *0.5000* | *0.2422* | *0.5000* | *0.2986* | *—* | *—* |

### 2.1 `eos − mean` 增益

| Prompt 组 | Δ frame AUC | Δ frame AP | Δ clip AUC | Δ clip AP |
|---|---:|---:|---:|---:|
| label | +0.0258 | +0.0120 | +0.0228 | +0.0229 |
| scene | **+0.1542** | **+0.0952** | **+0.1142** | **+0.0837** |
| contrast | +0.0717 | +0.0190 | +0.0383 | +0.0039 |
| all | +0.0834 | +0.0512 | +0.0835 | +0.0674 |

`eos` 在 4 组 × 4 指标 = 16 个组合中**全部**高于 `mean`。

---

## 3. 逐视频 frame AUC

| vid | frames | anomaly | label_mean | label_eos | scene_mean | scene_eos | contrast_mean | contrast_eos | all_mean | all_eos |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 01 | 1439 | 408 | 0.388 | 0.369 | 0.603 | 0.796 | 0.262 | 0.315 | 0.442 | 0.524 |
| 02 | 1211 | 89 | 0.487 | 0.590 | 0.535 | 0.529 | 0.478 | 0.444 | 0.511 | 0.555 |
| 03 | 923 | 87 | 0.617 | 0.600 | 0.637 | 0.479 | 0.363 | 0.394 | 0.637 | 0.562 |
| 04 | 947 | 93 | 0.422 | 0.433 | 0.523 | 0.545 | 0.117 | 0.175 | 0.432 | 0.457 |
| 05 | 1007 | 318 | 0.282 | 0.403 | 0.409 | 0.716 | 0.105 | 0.172 | 0.275 | 0.479 |
| 06 | 1283 | 429 | 0.537 | 0.330 | 0.510 | 0.744 | 0.291 | 0.268 | 0.471 | 0.461 |
| 07 | 605 | 123 | 0.288 | 0.212 | 0.270 | 0.174 | 0.046 | 0.147 | 0.227 | 0.168 |
| 08 | 36 | 36 | — | — | — | — | — | — | — | — |
| 09 | 1175 | 289 | 0.525 | 0.727 | 0.688 | 0.681 | 0.376 | 0.360 | 0.619 | 0.669 |
| 10 | 841 | 133 | 0.793 | 0.914 | 0.756 | 0.852 | 0.449 | 0.595 | 0.790 | 0.905 |
| 11 | 472 | 144 | 0.426 | 0.478 | 0.669 | 0.619 | 0.433 | 0.356 | 0.532 | 0.495 |
| 12 | 1271 | 190 | 0.291 | 0.338 | 0.375 | 0.556 | 0.202 | 0.217 | 0.329 | 0.421 |
| 13 | 549 | 75 | 0.188 | 0.192 | 0.443 | 0.835 | 0.209 | 0.295 | 0.285 | 0.435 |
| 14 | 507 | 74 | 0.467 | 0.649 | 0.744 | 0.803 | 0.340 | 0.391 | 0.576 | 0.709 |
| 15 | 1001 | 90 | 0.369 | 0.072 | 0.250 | 0.402 | 0.522 | 0.111 | 0.282 | 0.097 |
| 16 | 740 | 109 | 0.194 | 0.167 | 0.313 | 0.469 | 0.508 | 0.654 | 0.250 | 0.276 |
| 17 | 426 | 384 | 0.438 | 0.846 | 0.273 | 0.412 | 0.259 | 0.217 | 0.433 | 0.657 |
| 18 | 294 | 294 | — | — | — | — | — | — | — | — |
| 19 | 248 | 140 | 0.550 | 0.155 | 0.332 | 0.452 | 0.370 | 0.117 | 0.421 | 0.240 |
| 20 | 273 | 154 | 0.779 | 0.792 | 0.726 | 0.743 | 0.585 | 0.635 | 0.769 | 0.786 |
| 21 | 76 | 53 | 0.530 | 0.336 | 0.753 | 0.195 | 0.113 | 0.032 | 0.628 | 0.219 |

### 3.1 逐视频统计（排除全异常的 08、18，N=19）

| 组 | 均值 | 中位数 | AUC>0.5 | 最好 | 最差 |
|---|---:|---:|---:|---:|---:|
| label_mean | 0.451 | 0.438 | 7/19 | 0.793 | 0.188 |
| label_eos | 0.453 | 0.403 | 7/19 | 0.914 | 0.072 |
| scene_mean | 0.516 | 0.523 | 11/19 | 0.756 | 0.250 |
| **scene_eos** | **0.579** | **0.556** | **12/19** | 0.852 | 0.174 |
| contrast_mean | 0.317 | 0.340 | 3/19 | 0.585 | 0.046 |
| contrast_eos | 0.310 | 0.295 | 3/19 | 0.654 | 0.032 |
| all_mean | 0.469 | 0.442 | 8/19 | 0.790 | 0.227 |
| all_eos | 0.480 | 0.479 | 8/19 | 0.905 | 0.097 |

### 3.2 逐视频最优组

| vid | 最优组 (AUC) | vid | 最优组 (AUC) |
|---|---|---|---|
| 01 | scene_eos (0.796) | 12 | scene_eos (0.556) |
| 02 | label_eos (0.590) | 13 | scene_eos (0.835) |
| 03 | scene_mean (0.637) | 14 | scene_eos (0.803) |
| 04 | scene_eos (0.545) | 15 | contrast_mean (0.522) |
| 05 | scene_eos (0.716) | 16 | contrast_eos (0.654) |
| 06 | scene_eos (0.744) | 17 | label_eos (0.846) |
| 07 | label_mean (0.288) | 19 | label_mean (0.550) |
| 09 | label_eos (0.727) | 20 | label_eos (0.792) |
| 10 | label_eos (0.914) | 21 | scene_mean (0.753) |
| 11 | scene_mean (0.669) | | |

计数：scene 系 10/19，label 系 7/19，contrast 系 2/19。

---

## 4. 解释结果的 prompt 分布

每个视频排名第 1 的 prompt 统计：

| 组 | Top-1 prompt 分布 |
|---|---|
| scene_eos | `a suspicious crowd scene`：21/21 |
| label_eos | `a person shoplifting`：15/21；`vandalism`：4；`fighting`：2 |
| contrast_eos | `a video of suspicious movement`：18/21；`violent interaction`：3 |
| all_eos | `shoplifting`：11；`suspicious crowd scene`：7；`fighting`：2；`vandalism`：1 |

---

## 5. 关键观察

- 8 组中仅 `scene_eos`（frame AUC 0.649）与 `all_eos`（0.537）高于随机。
- `eos` 全面优于 `mean`，`scene` 提升最大（+0.154）。
- 组间排序（eos，frame AUC）：`scene 0.649 > all 0.537 > label 0.479 > contrast 0.373`。
- 31 条 prompt 的 `all` 低于 9 条的 `scene`（0.537 vs 0.649）。
- clip 指标普遍高于 frame 指标（clip 用 max 聚合且任务更易）。
- 全局 frame AUC（0.649）高于逐视频均值（0.579），两者口径不同。
- `video_auc/video_ap = 0.0` 是单类未定义（21 个视频全含异常），非性能为 0。
- 单视频方差大：`07` 最高仅 0.288，`15` 最低 0.072。
- Top-1 解释 prompt 高度集中（如 scene_eos 全为 `suspicious crowd scene`），解释区分度有限。

---

## 6. 复现

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# eval.text_pool = "mean"
python scripts/zero_shot_test.py --prompts label    --out-dir results/zero_shot_label_mean
python scripts/zero_shot_test.py --prompts scene    --out-dir results/zero_shot_scene_mean
python scripts/zero_shot_test.py --prompts contrast --out-dir results/zero_shot_contrast_mean
python scripts/zero_shot_test.py --prompts all      --out-dir results/zero_shot_all_mean

# eval.text_pool = "eos"
python scripts/zero_shot_test.py --prompts label    --out-dir results/zero_shot_label_eos
python scripts/zero_shot_test.py --prompts scene    --out-dir results/zero_shot_scene_eos
python scripts/zero_shot_test.py --prompts contrast --out-dir results/zero_shot_contrast_eos
python scripts/zero_shot_test.py --prompts all      --out-dir results/zero_shot_all_eos
```

每组输出：`summary.json`、`zero_shot_metrics.json`、`zero_shot_explanations.json`、
`zero_shot_plots/`。

---

## 7. 局限

- 零样本最好 frame AUC 0.649，仍不足以直接作为检测器，定位为微调基线。
- video-level 指标在当前测试集不可用（单类）。
- 单一 backbone，无时序建模；prompt 人工设计、极性为关键词规则。
- 解释为模板/检索式，且存在单一 prompt 支配。
