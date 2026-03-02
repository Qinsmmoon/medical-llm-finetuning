# Medical LLM Fine-tuning: LoRA vs DoRA Comparative Study

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## 📋 项目概述

本项目系统比较了两种高效的参数微调方法——LoRA (Low-Rank Adaptation) 和 DoRA (Weight-Decomposed Low-Rank Adaptation) 在医学问答任务上的表现。通过全面的实验设计、多维度的评估体系和深入的模型分析，探索了不同微调方法在专业领域应用的优缺点。

### 🎯 研究目标
- 系统对比 LoRA 和 DoRA 在医学领域的表现差异
- 探究不同超参数对模型性能的影响
- 分析参数效率与模型性能的平衡点
- 提供可落地的医学问答解决方案

## 🔬 主要发现

| 发现 | LoRA | DoRA | 结论 |
|------|------|------|------|
| 最佳秩(r) | 8 | 8 | r=8 在性能和参数间取得最佳平衡 |
| 小样本性能 | 中等 | **优秀** | DoRA 在数据稀缺时表现更好 |
| 训练稳定性 | 稳定 | **更稳定** | DoRA 训练曲线更平滑 |
| 参数效率 | **高** | 中等 | LoRA 参数量更少 |
| 推理速度 | **快** | 快 | 两者差异不大 |

## 🏗️ 项目结构

详见 [项目结构文档](docs/project_structure.md)

## 🚀 快速开始

### 环境配置

```bash
# 克隆仓库
git clone https://github.com/yourusername/medical-llm-finetuning.git
cd medical-llm-finetuning

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt