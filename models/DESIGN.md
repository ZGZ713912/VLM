# models/ 模块设计文档

## 1. 整体架构

```
video (B,T,C,H,W)                       prompts (list[str])
       │                                        │
       ▼                                        ▼
┌──────────────────┐                 ┌──────────────────┐
│  Vision Encoder  │                 │  Text Encoder    │
│  (CLIP ViT)      │                 │  (CLIP Text)     │
│  encode_video()  │                 │  encode_text()   │
└────────┬─────────┘                 └────────┬─────────┘
         │ (B, D)                            │ (B, D)
         │           ┌──────────┐             │
         └──────────►│  Fusion  │◄────────────┘
                     │  (2种)   │
                     └────┬─────┘
                          │ (B, D_f)
                          ▼
                     ┌──────────┐
                     │   Head   │
                     └────┬─────┘
                          │
                          ▼
                     {
                       anomaly_score: (B,),
                       frame_score:   (B, T),
                       embedding:     (B, D_f)
                     }
```

---

## 2. 文件拆分为三个

| 文件 | 职责 | 核心类 |
|---|---|---|
| `backbone.py` | Vision + Text 编码器，冻结/微调管理 | `CLIPBackbone` |
| `fusion.py` | 视觉-文本特征融合 | `ConcatFusion`, `CrossAttnFusion` |
| `head.py` | 异常分数预测 | `AnomalyHead` |

顶层 `models/__init__.py` 提供一个 `VLMModel` 把三者拼成完整模型。

---

## 3. backbone.py — 编码器

### 3.1 设计目标

- 基于 CLIP：vision encoder 和 text encoder 已预对齐
- 两者放在同一个类里，统一管理 `requires_grad_()`
- `encode_video()` 和 `encode_text()` 独立调用，下游决定怎么组合
- 不绑定死 CLIP — model_name 可换，未来也可能替换整个 encoder

### 3.2 接口

```python
class CLIPBackbone(nn.Module):
    def __init__(self, model_name="ViT-B-32", pretrained="laion2b_s34b_b79k"):
        ...

    def encode_video(self, video: Tensor) -> Tensor:
        """
        video: (B, T, C, H, W)
        return: (B, D)   归一化后的视觉特征
        """

    def encode_text(self, prompts: list[str]) -> Tensor:
        """
        prompts: ["a person running", ...]
        return: (B, D)   归一化后的文本特征
        """

    @property
    def dim(self) -> int:
        """输出维度，方便下游模块自动适配"""
```

### 3.3 encode_video 内部流程

```
(B, T, C, H, W)
    │ reshape
    ▼
(B*T, C, H, W)         # 把时间维并入 batch 维
    │ self.model.encode_image()
    ▼
(B*T, D)               # CLIP 逐帧编码
    │ reshape
    ▼
(B, T, D)              # 恢复时间维
    │ mean(dim=1)
    ▼
(B, D)                 # 时序平均池化 → clip 级特征
    │ L2 normalize
    ▼
(B, D)                 # 最终输出
```

**为什么用 mean pooling 而不是取最后一帧或 flatten：**
- 视频异常往往在整个 clip 的时间维度上有分布
- mean 是最简单、最不引入额外参数的聚合方式
- 如果需要更强的时序建模，未来可以在 fusion 层做（例如加 position embedding + Transformer），不影响 backbone

### 3.4 encode_text 内部流程

```
["prompt 1", "prompt 2", ...]
    │ self.tokenizer()
    ▼
(B, 77)                # token ids，padding 到 77
    │ self.model.encode_text()
    ▼
(B, D)                 # CLIP 文本编码
    │ L2 normalize
    ▼
(B, D)                 # 最终输出
```

### 3.5 冻结/微调管理

PyTorch 原生支持，不需要额外封装：

```python
# 冻结视觉，只训练文本
backbone.vision_encoder.requires_grad_(False)

# 冻结文本，只训练视觉
backbone.text_encoder.requires_grad_(False)

# 都冻结（只用预训练特征）
backbone.requires_grad_(False)

# 都微调
backbone.requires_grad_(True)
```

注意：CLIP 的 `encode_image` 和 `encode_text` 在 `self.model.visual` 和 `self.model.transformer`（或类似属性）上，需要暴露这两个子模块为属性。

---

## 4. fusion.py — 特征融合

### 4.1 设计目标

- 至少支持 concat fusion 和 cross-attention fusion（AGENTS.md 要求）
- 输入两个 (B, D) 向量，输出一个 (B, D_f) 融合向量
- 可视化/文本对齐信息通过融合层传递

### 4.2 ConcatFusion

```python
class ConcatFusion(nn.Module):
    """
    visual: (B, D)  +  text: (B, D)
        ↓ concat
    (B, 2*D)
        ↓ Linear + ReLU
    (B, D_f)
    """
```

最简融合方式。适合作为 baseline。

### 4.3 CrossAttnFusion

```python
class CrossAttnFusion(nn.Module):
    """
    visual 做 Query，text 做 Key/Value：
    
    Q = Linear_q(visual)   # (B, D) → (B, d_k)
    K = Linear_k(text)     # (B, D) → (B, d_k)  
    V = Linear_v(text)     # (B, D) → (B, d_v)
    
    attn = softmax(Q @ K.T / sqrt(d_k))    # (B, B)
    output = attn @ V                       # (B, d_v)
    
    最后 Linear → (B, D_f)
    """
```

视觉特征"关注"文本的哪些部分。更灵活，但参数更多。

**单头还是多头：** 先用单头（单层 cross-attention），多头会增加复杂度但对 batch 内样本间注意的收益有限。后续实验需要时再加。

### 4.4 选哪个

通过在 `VLMModel.__init__` 里传参切换：

```python
model = VLMModel(backbone=backbone, fusion="concat", head=head)
model = VLMModel(backbone=backbone, fusion="cross_attn", head=head)
```

---

## 5. head.py — 异常检测头

### 5.1 设计目标

- 输入融合特征 (B, D_f)，输出异常分数
- 需要同时支持 clip 级和帧级预测
- 简单、可扩展

### 5.2 接口

```python
class AnomalyHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 256):
        ...

    def forward(self, fused: Tensor) -> dict:
        """
        fused: (B, D_f)
        return: {
            "anomaly_score": (B,),       # clip 级异常概率 0~1
            "frame_score":   (B, T),     # 帧级异常概率（暂时从 clip 级复制）
            "embedding":     (B, D_f),   # 融合特征，供对比 loss 使用
        }
        """
```

### 5.3 结构

```
(B, D_f)
    │
    ├─→ Linear → ReLU → Dropout → Linear → Sigmoid → anomaly_score (B,)
    │
    └─→ embedding (B, D_f)   # 直接透传，用于 contrastive loss
```

frame_score 第一版从 anomaly_score 扩展：`frame_score = anomaly_score.unsqueeze(1).expand(-1, T)`。

与 AGENTS.md `explanation` 字段的关系：第一版不实现 explanation，只返回 `anomaly_score` / `frame_score` / `embedding`。explanation 留到后续迭代。

---

## 6. models/__init__.py — 顶层拼装

```python
class VLMModel(nn.Module):
    def __init__(self, backbone, fusion, head):
        self.backbone = backbone
        self.fusion   = fusion
        self.head     = head

    def forward(self, video, prompts):
        """
        video:   (B, T, C, H, W)
        prompts: list[str]  or  (B, D) 预编码的 text embedding

        return: {
            anomaly_score: (B,),
            frame_score:   (B, T),
            embedding:     (B, D_f),
        }
        """
```

forward 内部流程：

```
1. vis_emb = backbone.encode_video(video)          # (B, D)
2. if isinstance(prompts, Tensor):
       txt_emb = prompts                            # 预编码
   else:
       txt_emb = backbone.encode_text(prompts)      # 现场编码
3. fused = fusion(vis_emb, txt_emb)                 # (B, D_f)
4. output = head(fused)                             # dict
5. return output
```

**兼容方案 B**：如果传预计算的 text embedding，跳过 `encode_text()`；如果传字符串列表，跑 `encode_text()`。两者都支持。

---

## 7. 数据流完整示例

```python
from data import build_avenue_dataset, build_avenue_dataloader
from models import VLMModel
from models.backbone import CLIPBackbone
from models.fusion import ConcatFusion
from models.head import AnomalyHead

# 1. 构建数据
loader = build_avenue_dataloader(root="./data", split="training", batch_size=8)
batch = next(iter(loader))
video   = batch["video"]       # (8, 16, 3, 224, 224)
prompts = batch["prompt_text"] # list of 8 strings

# 2. 构建模型
backbone = CLIPBackbone(model_name="ViT-B-32", pretrained="laion2b_s34b_b79k")
fusion   = ConcatFusion(in_dim=backbone.dim, out_dim=512)
head     = AnomalyHead(in_dim=512, hidden_dim=256)
model    = VLMModel(backbone=backbone, fusion=fusion, head=head)

# 3. 冻结 backbone，只训练 fusion + head
backbone.requires_grad_(False)

# 4. forward
output = model(video, prompts)
# output["anomaly_score"] shape: (8,)
# output["frame_score"]   shape: (8, 16)
# output["embedding"]     shape: (8, 512)
```

---

## 8. 三层之间的输入输出约定

| 层级 | 输入 | 输出 |
|---|---|---|
| `backbone.encode_video` | `(B,T,C,H,W)` | `(B, D)` |
| `backbone.encode_text` | `list[str]` | `(B, D)` |
| `fusion` | `(B, D)` + `(B, D)` | `(B, D_f)` |
| `head` | `(B, D_f)` | `{anomaly_score, frame_score, embedding}` |

关键约束：`backbone` 输出的视觉和文本向量必须在**相同维度 D**（CLIP 保证这一点），且 **L2 归一化**（利于 cosine similarity 和 contrastive loss）。

---

## 9. 不做什么

| 不做 | 原因 |
|---|---|
| `explanation` 字段 | 先跑通基础 pipeline，后续迭代加 |
| 多头 cross-attention | 先单头验证效果，需要时再加 |
| 其他 backbone（Swin, ViT+BERT） | 先用 CLIP 跑通，接口够通用，后续替换容易 |
| `frame_score` 精细预测 | 第一版从 clip 级复制，后续可用时序 head |
| tokenizer 放在 prompts/ | tokenizer 和 text encoder 强绑定，拆开会增加传递复杂度 |
