# VLM-WS: Vision-Language Model for Video Anomaly Detection

基于视觉语言模型（VLM）+ Prompt Engineering 的视频异常检测与可解释性研究项目。

> **核心思想：** Vision-Language Model + Prompt Engineering → Video Anomaly Detection + Explainability

## 前置要求

| 依赖 | 说明 |
|---|---|
| Docker Engine 24+ | 容器运行时 |
| Docker Compose v2 | 多容器编排 |
| NVIDIA Container Toolkit | **可选**，仅 GPU 训练需要 |

## 快速启动

```bash
# 1. 克隆仓库
git clone <repo-url>
cd vlm_ws

# 2. 创建环境变量文件
cp .env.example .env

# 3. 构建镜像（首次约 5-10 分钟）
docker compose build

# 4. 启动容器
docker compose up -d

# 5. 进入开发环境
docker compose exec vlm-dev zsh
```

### GPU 主机

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d
```

### CPU 主机

在 `.env` 中切换为 CPU 镜像：

```bash
sed -i 's|^BASE_IMAGE=.*|BASE_IMAGE=pytorch/pytorch:2.3.1-cpu|' .env
docker compose build
```

## 数据集

项目使用 [CUHK Avenue Dataset](http://www.cse.cuhk.edu.hk/leojia/projects/detectabnormal/dataset.html) 进行视频异常检测实验。

下载后将数据集放入 `data/` 目录，结构如下：

```
data/
├── __init__.py
├── avenue_dataset.py
├── dataloader.py
└── Avenue_Dataset/          # 从官网下载后解压到这里
    ├── training_videos/     # 16 个训练视频 (.avi)
    ├── testing_videos/      # 21 个测试视频 (.avi)
    ├── training_vol/        # 16 个像素级标注 (.mat)
    └── testing_vol/         # 21 个像素级标注 (.mat)
```

> 数据集文件被 `.gitignore` 排除，不会提交到仓库。

## 项目结构

```
vlm_ws/
├── models/                  # 模型模块
│   ├── DESIGN.md            # 模型架构设计文档
│   ├── __init__.py          # VLMModel（backbone + fusion + head）
│   ├── backbone.py          # CLIP 视觉 + 文本编码器
│   └── fusion.py            # ConcatFusion / CrossAttnFusion
│   └── head.py              # 异常检测头
├── data/                    # 数据模块
│   ├── avenue_dataset.py    # Avenue 数据集类
│   └── dataloader.py        # DataLoader 工厂函数
├── prompts/                 # Prompt 模板管理
│   └── __init__.py          # label / scene / contrast 三种模板
├── train/                   # 训练循环
├── eval/                    # 评估指标
├── utils/                   # 日志、可视化、可复现工具
├── configs/                 # 实验配置文件
├── checkpoints/             # 模型权重（运行时生成）
├── logs/                    # TensorBoard 日志（运行时生成）
├── results/                 # 实验结果（运行时生成）
├── requirements/            # Python 依赖
│   ├── base.txt             # 核心依赖
│   └── dev.txt              # 开发依赖（Jupyter、pytest 等）
├── scripts/docker/          # 容器入口脚本与 shell 模版
├── Dockerfile               # 镜像构建文件
├── docker-compose.yml       # 开发容器编排
├── docker-compose.gpu.yml   # GPU 覆盖配置
├── .env.example             # 环境变量模板
├── .devcontainer/           # VS Code Dev Container 配置
├── docs/container.md        # 容器环境详细文档
└── AGENTS.md                # AI Agent 编码规范
```

## 容器内开发

```bash
# Jupyter Lab（端口 8888）
jupyter lab --ip 0.0.0.0 --port 8888 --no-browser

# TensorBoard（端口 6006）
tensorboard --logdir logs --host 0.0.0.0 --port 6006
```

## 环境变量

| 变量 | 默认值 | 说明 |
|---|---|---|
| `BASE_IMAGE` | `pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime` | 基础镜像 |
| `LOCAL_UID:GID` | `1000:1000` | 容器内用户 ID |
| `SHM_SIZE` | `16gb` | 共享内存大小 |
| `JUPYTER_PORT` | `8888` | Jupyter 端口映射 |
| `TENSORBOARD_PORT` | `6006` | TensorBoard 端口映射 |

在 `.env` 文件中修改即可。

## 技术栈

- **框架：** PyTorch 2.3, OpenCLIP, HuggingFace Transformers
- **视频处理：** OpenCV, Decord
- **语言模型：** CLIP (ViT + Text Transformer)
- **容器：** Docker, Docker Compose
- **依赖注入：** OmegaConf, PyYAML

## 更多文档

- [容器环境详细说明](docs/container.md)
- [模型架构设计](models/DESIGN.md)
- [AI Agent 编码规范](AGENTS.md)
