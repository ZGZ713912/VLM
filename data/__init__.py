# 这个文件是 data 包的统一导出入口。
"""Dataset and dataloader modules."""

# 导出 AvenueDataset 类和默认 prompt 模板常量。
from .avenue_dataset import AvenueDataset, DEFAULT_PROMPT_TEMPLATES
# 导出 dataloader 构建函数和 worker 随机种子函数。
from .dataloader import build_avenue_dataloader, build_avenue_dataset, seed_worker

# __all__ 用来声明“from data import *”时允许暴露哪些名字。
__all__ = [
    "AvenueDataset",  # Avenue 数据集类。
    "DEFAULT_PROMPT_TEMPLATES",  # 默认 prompt 模板字典。
    "build_avenue_dataloader",  # 构建 DataLoader 的工厂函数。
    "build_avenue_dataset",  # 构建 Dataset 的工厂函数。
    "seed_worker",  # 给 dataloader worker 设置随机种子的函数。
]
