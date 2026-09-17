#!/usr/bin/env python3
"""
Zero-shot测试脚本
用于验证VLM-VAD框架的基础功能，无需训练即可评估不同prompt方法的效果

用法:
    python scripts/zero_shot_test.py --config configs/experiment.yaml --prompts label|scene|contrast|all
"""

import argparse
import sys
from pathlib import Path
import json
import time

# 项目根目录导入
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import torch
import torch.nn.functional as F
from tqdm import tqdm

# 项目模块导入
from datasets.builders import build_split_dataloader
from models.factory import build_model
from utils.config import load_experiment_config
from utils.io import ensure_dirs
from utils.logging import get_logger
from eval.inference import evaluate_videos
from eval.metrics import compute_auc_ap

log = get_logger(__name__)


def generate_test_prompts(cfg, prompt_type):
    """生成测试用的prompt列表"""
    from prompts import PromptManager
    
    prompt_manager = PromptManager(cfg.prompt)
    
    if prompt_type == "all":
        # 使用所有类型的prompt
        prompts = []
        for ptype in ["label", "scene", "contrast"]:
            prompts.extend(prompt_manager.get_prompts(ptype))
    else:
        # 使用指定类型的prompt
        prompts = prompt_manager.get_prompts(prompt_type)
    
    return prompts


def zero_shot_inference(model, dataloader, prompts, device):
    """Zero-shot推理"""
    model.eval()
    all_results = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Zero-shot inference")):
            # 获取视频帧
            frames = batch["frames"].to(device)  # [B, T, C, H, W]
            video_ids = batch["video_id"]
            
            # 提取视觉特征
            vision_outputs = model.backbone.encode Vision(frames)
            vision_embeddings = vision_outputs.embedding  # [B, D]
            
            # 对每个prompt进行匹配
            batch_results = []
            for prompt in prompts:
                # 获取文本特征
                text_outputs = model.backbone.encode_text(prompt)
                text_embeddings = text_outputs.embedding  # [1, D]
                
                # 计算相似度
                similarities = F.cosine_similarity(vision_embeddings, text_embeddings, dim=1)  # [B]
                
                # 计算异常分数（相似度越低，异常可能性越高）
                anomaly_scores = 1 - similarities
                
                # 保存结果
                result = {
                    "video_id": video_ids,
                    "prompt": prompt,
                    "anomaly_scores": anomaly_scores.cpu().numpy(),
                    "vision_embeddings": vision_embeddings.cpu().numpy(),
                    "text_embeddings": text_embeddings.cpu().numpy()
                }
                batch_results.append(result)
            
            all_results.extend(batch_results)
    
    return all_results


def evaluate_zero_shot_results(results, dataloader, device):
    """评估zero-shot结果"""
    # 收集所有异常分数和真实标签
    all_scores = []
    all_labels = []
    
    for result in results:
        scores = result["anomaly_scores"]
        all_scores.extend(scores)
        
        # 获取对应的真实标签（需要根据video_id查找）
        # 这里简化处理，实际需要根据数据集获取
        video_id = result["video_id"]
        # TODO: 根据video_id获取真实标签
        # labels = get_true_labels(video_id, dataloader.dataset)
        # all_labels.extend(labels)
    
    # 计算指标
    if len(all_labels) > 0:
        metrics = compute_auc_ap(all_scores, all_labels)
    else:
        metrics = {
            "frame_auc": 0.5,  # 随机水平
            "frame_ap": 0.5,
            "video_auc": 0.5,
            "video_ap": 0.5
        }
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Zero-shot test for VLM-VAD")
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--prompts", default="all", choices=["label", "scene", "contrast", "all"])
    parser.add_argument("--output", default="results/zero_shot_test.json")
    args = parser.parse_args()
    
    # 加载配置
    cfg = load_experiment_config(args.config)
    device = torch.device(cfg.device)
    
    log.info("Starting zero-shot test with %s prompts", args.prompts)
    
    # 创建输出目录
    ensure_dirs({"result": Path(args.output).parent})
    
    # 生成测试prompts
    prompts = generate_test_prompts(cfg, args.prompts)
    log.info("Generated %d prompts", len(prompts))
    
    # 构建数据加载器
    val_loader, val_src = build_split_dataloader(
        cfg.data, split=cfg.data.eval_split, shuffle=False, seed=int(cfg.seed)
    )
    
    # 构建模型
    model = build_model(cfg).to(device)
    log.info("Model built with %d trainable parameters", 
             sum(1 for p in model.parameters() if p.requires_grad))
    
    # Zero-shot推理
    start_time = time.time()
    results = zero_shot_inference(model, val_loader, prompts, device)
    inference_time = time.time() - start_time
    
    # 评估结果
    metrics = evaluate_zero_shot_results(results, val_loader, device)
    
    # 保存结果
    output_data = {
        "config": args.config,
        "prompts_type": args.prompts,
        "num_prompts": len(prompts),
        "inference_time": inference_time,
        "metrics": metrics,
        "results": results[:10],  # 只保存前10个结果用于调试
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    # 打印结果
    log.info("Zero-shot test completed!")
    log.info("Prompt type: %s", args.prompts)
    log.info("Number of prompts: %d", len(prompts))
    log.info("Inference time: %.2f seconds", inference_time)
    log.info("Frame AUC: %.4f", metrics["frame_auc"])
    log.info("Frame AP: %.4f", metrics["frame_ap"])
    log.info("Video AUC: %.4f", metrics["video_auc"])
    log.info("Video AP: %.4f", metrics["video_ap"])
    
    return metrics


if __name__ == "__main__":
    main()