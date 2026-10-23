# 区域与时序正常性训练（2026-10-23）

目标是检测偏离监控场景正常模式的帧，并提供异常帧排名、区域与候选文本说明。
旧分类器只学运动伪标签；冻结 CLIP 的中心裁剪丢失宽画面两侧，整帧特征又稀释小目标。
本阶段先固定协议与模块，再训练和查看测试成绩。

## 数据与标注

使用已有 Avenue 和自行下载的 UCSD Ped2。Ped2 从 UCSD 官方归档获取，记录 URL、
SHA256、文件与帧数；仅提取 Ped2。官方 TIFF/BMP 异常 mask 按任意非零像素派生帧标签，
与原始 MATLAB 帧区间交叉验证。标注是发布方人工标注的转换，不声称重新人工标注。
所有 training 视频按正常训练协议使用，按完整视频固定划分拟合、校准；测试真值只计分。

## 模块与张量

- `datasets/region_features.py`：全幅 RGB squash 到编码器输入，CLIP ViT patch token
  投影为 `(N,R,D)`，全局特征 `(N,D)`；同网格 Farneback 光流描述 `(N,R,F)`。
  不将运动强弱当作异常真值。缓存核验编码器、变换、源文件哈希与输出哈希。
- `tools/prepare_ped2.py`：可恢复下载、安全提取、逐帧 mask/区间验证和无损 AVI 转换。
- `models/region_temporal.py`：冻结视觉/文本编码器后的轻量区域时序预测器，
  支持 concat/crossattn；输入区域 `(B,T,R,D)`、运动 `(B,T,R,F)`、外部文本 `(K,D)`；
  输出 anomaly_score `(B,)`、frame_score `(B,T)`、embedding `(B,D)`、explanation，
  以及分量用于解释。正常参考库补充局部外观与运动新颖度。
- `train/region_trainer.py` 与训练入口：正常样本的下一帧预测；正常/合成扰动 BCE；正常视觉-文本
  对比对齐。合成扰动标签只监督“是否人为扰动”，不当作真实异常行为标签。
- `eval/region_evaluation.py`：仅校准训练视频固定各分量尺度和阈值；完整视频输出，
  pooled raw-score AUC/AP，不逐测试视频归一化；保留逐视频与聚类 bootstrap 置信区间。
- `scripts/train_region_model.py` / `scripts/run_region_experiments.py`：配置化训练、
  resume、prompt/fusion/backbone/seed 矩阵、冻结模型后独立评估。

## 固定实验规则

首轮使用 ViT-B-32/laion2b_s34b_b79k；第二骨干按可用公开权重配置，缓存严格隔离。
每个数据集使用相同的正常训练协议。epoch 只按留出 training 视频预测误差选取，
各分量权重、平滑窗口与阈值在查看本轮测试结果前写入配置；同时报告外观、运动、
预测和融合消融，禁止按照测试成绩挑权重。prompt 类型显式 label/scene/contrast，
文本在 prompts 配置中，不在 forward 内生成。

验收至少包含：真实训练与独立官方计分；对比原有同协议基线；缓存/标签错误拒绝；
训练恢复确定性；视频隔离；预测无未来信息泄漏；训练和测试解释不过度声称事件语义。
如成绩未提升，保留失败并据训练/校准证据修订下一假设，不用反复试测试集挑参。

已完成实验见 [region_temporal_results.md](region_temporal_results.md)。
