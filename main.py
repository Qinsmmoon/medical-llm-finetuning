import os
import torch
import pandas as pd
from datasets import Dataset
from config.training_config import TrainingConfig
from data.processor import format_dataset_jsonl, preprocess_function
from models.loader import load_model_and_tokenizer
from models.inference import QwenInference
from models.judge import JudgeEvaluator
from training.trainer import train_model
from utils.logger import setup_logger

def main():
    logger = setup_logger()
    logger.info("=" * 60)
    logger.info("Qwen3-1.7B LoRA/DoRA 微调及 LLM-as-a-Judge 对比")
    logger.info("=" * 60)

    # 创建必要的目录
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs(TrainingConfig.OUTPUT_BASE_DIR, exist_ok=True)

    # 准备数据集
    logger.info("\n[步骤1] 准备数据集...")
    train_formatted_path = "data/processed/train_formatted.jsonl"
    val_formatted_path = "data/processed/val_formatted.jsonl"
    
    if not os.path.exists(train_formatted_path):
        format_dataset_jsonl(TrainingConfig.TRAIN_DATA_PATH, train_formatted_path)
    if not os.path.exists(val_formatted_path):
        format_dataset_jsonl(TrainingConfig.VAL_DATA_PATH, val_formatted_path)

    # 加载数据集
    train_df = pd.read_json(train_formatted_path, lines=True)
    val_df = pd.read_json(val_formatted_path, lines=True)
    train_ds = Dataset.from_pandas(train_df)
    val_ds = Dataset.from_pandas(val_df)

    # 预分词
    from transformers import AutoTokenizer
    temp_tokenizer = AutoTokenizer.from_pretrained(
        TrainingConfig.MODEL_NAME,
        cache_dir=TrainingConfig.MODEL_CACHE_DIR,
        use_fast=False,
        trust_remote_code=True
    )
    if temp_tokenizer.pad_token is None:
        temp_tokenizer.pad_token = temp_tokenizer.eos_token

    train_dataset = train_ds.map(
        lambda x: preprocess_function(x, temp_tokenizer),
        remove_columns=train_ds.column_names,
        batched=False
    )
    eval_dataset = val_ds.map(
        lambda x: preprocess_function(x, temp_tokenizer),
        remove_columns=val_ds.column_names,
        batched=False
    )

    # 训练 LoRA 模型
    logger.info("\n[步骤2] 训练 LoRA 模型...")
    lora_output_dir = os.path.join(TrainingConfig.OUTPUT_BASE_DIR, "lora")
    os.makedirs(lora_output_dir, exist_ok=True)
    model_lora, tokenizer_lora = load_model_and_tokenizer(use_dora=False)
    train_model(model_lora, tokenizer_lora, train_dataset, eval_dataset, lora_output_dir)
    del model_lora, tokenizer_lora
    torch.cuda.empty_cache()

    # 训练 DoRA 模型
    logger.info("\n[步骤3] 训练 DoRA 模型...")
    dora_output_dir = os.path.join(TrainingConfig.OUTPUT_BASE_DIR, "dora")
    os.makedirs(dora_output_dir, exist_ok=True)
    model_dora, tokenizer_dora = load_model_and_tokenizer(use_dora=True)
    train_model(model_dora, tokenizer_dora, train_dataset, eval_dataset, dora_output_dir)
    del model_dora, tokenizer_dora
    torch.cuda.empty_cache()

    # LLM-as-a-Judge 对比
    logger.info("\n[步骤4] 进行 LLM-as-a-Judge 对比...")
    test_questions = val_df["input"].tolist()[:50]
    
    # 生成回答
    base_infer = QwenInference(base_model_name='./autodl-tmp/Qwen/Qwen3-1.7B')
    base_responses = [base_infer.generate_response(q) for q in test_questions]

    lora_infer = QwenInference(
        base_model_name='./autodl-tmp/Qwen/Qwen3-1.7B',
        lora_adapter_path=os.path.join(lora_output_dir, "final_adapter")
    )
    lora_responses = [lora_infer.generate_response(q) for q in test_questions]

    dora_infer = QwenInference(
        base_model_name='./autodl-tmp/Qwen/Qwen3-1.7B',
        lora_adapter_path=os.path.join(dora_output_dir, "final_adapter")
    )
    dora_responses = [dora_infer.generate_response(q) for q in test_questions]

    # 裁判评估
    judge = JudgeEvaluator()
    judge_results = judge.evaluate(test_questions, base_responses, lora_responses, dora_responses)

    # 输出结果
    logger.info("\n" + "=" * 60)
    logger.info("LLM-as-a-Judge 对比结果")
    judge.print_results(judge_results)

    # 保存结果
    judge.save_results(
        judge_results, 
        test_questions, 
        base_responses, 
        lora_responses, 
        dora_responses,
        os.path.join(TrainingConfig.OUTPUT_BASE_DIR, "judge_results.json")
    )

    logger.info("\n全部流程完成！")

if __name__ == "__main__":
    main()