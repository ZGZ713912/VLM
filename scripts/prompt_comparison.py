#!/usr/bin/env python3
"""
Prompt对比实验脚本
比较不同prompt类型和组合的效果

用法:
    python scripts/prompt_comparison.py --config configs/experiment.yaml
"""

import argparse
import sys
from pathlib import Path
import json
import time
from typing import Dict, List, Any

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
from eval.metrics import compute_auc_ap

log = get_logger(__name__)


class PromptComparator:
    """Prompt对比器"""
    
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.model = build_model(cfg).to(self.device)
        self.prompt_manager = self._init_prompt_manager()
        
    def _init_prompt_manager(self):
        """初始化prompt管理器"""
        from prompts import PromptManager
        return PromptManager(self.cfg.prompt)
    
    def generate_experiment_prompts(self, experiment_name: str) -> List[str]:
        """生成指定实验的prompt列表"""
        config = self.cfg.prompt_experiment_configs[experiment_name]
        prompts = []
        
        for ptype in config["types"]:
            type_prompts = self.prompt_manager.get_prompts(ptype, count=config["count"])
            prompts.extend(type_prompts)
        
        return prompts
    
    def run_zero_shot_experiment(self, experiment_name: str, dataloader) -> Dict[str, Any]:
        """运行单个zero-shot实验"""
        log.info("Running experiment: %s", experiment_name)
        
        # 生成prompts
        prompts = self.generate_experiment_prompts(experiment_name)
        log.info("Generated %d prompts for %s", len(prompts), experiment_name)
        
        # Zero-shot推理
        results = self._zero_shot_inference(dataloader, prompts)
        
        # 评估结果
        metrics = self._evaluate_results(results, dataloader)
        
        return {
            "experiment_name": experiment_name,
            "num_prompts": len(prompts),
            "prompts": prompts,
            "metrics": metrics,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
    
    def _zero_shot_inference(self, dataloader, prompts: List[str]) -> List[Dict]:
        """Zero-shot推理"""
        self.model.eval()
        all_results = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="Zero-shot inference")):
                frames = batch["frames"].to(self.device)
                video_ids = batch["video_id"]
                
                # 提取视觉特征
                vision_outputs = self.model.backbone.encode_vision(frames)
                vision_embeddings = vision_outputs.embedding
                
                # 对每个prompt进行匹配
                batch_results = []
                for prompt in prompts:
                    # 获取文本特征
                    text_outputs = self.model.backbone.encode_text(prompt)
                    text_embeddings = text_outputs.embedding
                    
                    # 计算相似度和异常分数
                    similarities = F.cosine_similarity(vision_embeddings, text_embeddings, dim=1)
                    anomaly_scores = 1 - similarities
                    
                    result = {
                        "video_id": video_ids,
                        "prompt": prompt,
                        "anomaly_scores": anomaly_scores.cpu().numpy(),
                        "vision_embeddings": vision_embeddings.cpu().numpy()
                    }
                    batch_results.append(result)
                
                all_results.extend(batch_results)
        
        return all_results
    
    def _evaluate_results(self, results: List[Dict], dataloader) -> Dict[str, float]:
        """评估实验结果"""
        # 收集所有分数和标签
        all_scores = []
        all_labels = []
        
        # TODO: 实现真实的标签获取逻辑
        # 这里简化处理，实际需要根据数据集获取真实标签
        for result in results:
            scores = result["anomaly_scores"]
            all_scores.extend(scores)
            
            # 根据video_id获取真实标签
            video_id = result["video_id"]
            # labels = get_true_labels(video_id, dataloader.dataset)
            # all_labels.extend(labels)
        
        # 计算指标
        if len(all_labels) > 0:
            metrics = compute_auc_ap(all_scores, all_labels)
        else:
            # 如果没有真实标签，返回随机水平
            metrics = {
                "frame_auc": 0.5,
                "frame_ap": 0.5,
                "video_auc": 0.5,
                "video_ap": 0.5
            }
        
        return metrics


def main():
    parser = argparse.ArgumentParser(description="Prompt comparison experiment")
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--output", default="results/prompt_comparison.json")
    parser.add_argument("--experiments", nargs="+", 
                       default=["label_only", "scene_only", "contrast_only", "all_types"])
    args = parser.parse_args()
    
    # 加载配置
    cfg = load_experiment_config(args.config)
    device = torch.device(cfg.device)
    
    log.info("Starting prompt comparison experiment")
    
    # 创建输出目录
    ensure_dirs({"result": Path(args.output).parent})
    
    # 构建数据加载器
    val_loader, val_src = build_split_dataloader(
        cfg.data, split=cfg.data.eval_split, shuffle=False, seed=int(cfg.seed)
    )
    
    # 创建对比器
    comparator = PromptComparator(cfg)
    
    # 运行所有实验
    all_results = []
    start_time = time.time()
    
    for experiment_name in args.experiments:
        try:
            result = comparator.run_zero_shot_experiment(experiment_name, val_loader)
            all_results.append(result)
        except Exception as e:
            log.error("Experiment %s failed: %s", experiment_name, str(e))
            continue
    
    total_time = time.time() - start_time
    
    # 保存结果
    output_data = {
        "config": args.config,
        "experiments": args.experiments,
        "total_time": total_time,
        "results": all_results,
        "summary": _generate_summary(all_results),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    # 打印总结
    _print_summary(all_results)
    
    return output_data


def _generate_summary(results: List[Dict]) -> Dict[str, Any]:
    """生成实验总结"""
    summary = {
        "best_experiment": None,
        "best_auc": 0.0,
        "best_ap": 0.0,
        "avg_auc": 0.0,
        "avg_ap": 0.0,
        "experiment_count": len(results)
    }
    
    if not results:
        return summary
    
    # 计算平均性能
    total_auc = sum(r["metrics"]["frame_auc"] for r in results)
    total_ap = sum(r["metrics"]["frame_ap"] for r in results)
    summary["avg_auc"] = total_auc / len(results)
    summary["avg_ap"] = total_ap / len(results)
    
    # 找到最佳实验
    for result in results:
        metrics = result["metrics"]
        if metrics["frame_auc"] > summary["best_auc"]:
            summary["best_auc"] = metrics["frame_auc"]
            summary["best_experiment"] = result["experiment_name"]
        if metrics["frame_ap"] > summary["best_ap"]:
            summary["best_ap"] = metrics["frame_ap"]
            summary["best_experiment"] = result["experiment_name"]
    
    return summary


def _print_summary(results: List[Dict]):
    """打印实验总结"""
    log.info("=" * 60)
    log.info("PROMPT COMPARISON EXPERIMENT SUMMARY")
    log.info("=" * 60)
    
    if not results:
        log.error("No experiments completed successfully")
        return
    
    # 打印每个实验的结果
    for result in results:
        name = result["experiment_name"]
        metrics = result["metrics"]
        num_prompts = result["num_prompts"]
        
        log.info("Experiment: %s (%d prompts)", name, num_prompts)
        log.info("  Frame AUC: %.4f", metrics["frame_auc"])
        log.info("  Frame AP:  %.4f", metrics["frame_ap"])
        log.info("  Video AUC: %.4f", metrics["video_auc"])
        log.info("  Video AP:  %.4f", metrics["video_ap"])
        log.info("")
    
    # 打印总结
    summary = _generate_summary(results)
    log.info("SUMMARY:")
    log.info("Best Experiment: %s", summary["best_experiment"])
    log.info("Best Frame AUC:  %.4f", summary["best_auc"])
    log.info("Best Frame AP:   %.4f", summary["best_ap"])
    log.info("Average AUC:     %.4f", summary["avg_auc"])
    log.info("Average AP:      %.4f", summary["avg_ap"])
    log.info("=" * 60)


if __name__ == "__main__":
    main()