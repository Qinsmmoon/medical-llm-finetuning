import json
import os
import torch
import pandas as pd
from datasets import Dataset
from modelscope import snapshot_download
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer, 
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, TaskType, get_peft_model
import swanlab

# ==================== 1. 配置区域 ====================
class TrainingConfig:
    MODEL_NAME = "Qwen/Qwen3-1.7B"
    MODEL_CACHE_DIR = "./autodl-tmp/"
    
    # 本地数据路径 (请确保路径正确)
    TRAIN_JSON_PATH = "data/CMB/CMB-Exam/CMB-train/CMB-train-merge.json"
    VAL_JSON_PATH = "data/CMB/CMB-Exam/CMB-val/CMB-val-merge.json"
    
    # 训练超参
    LORA_R = 8
    LORA_ALPHA = 32
    LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    
    BATCH_SIZE = 4
    GRADIENT_ACCUMULATION_STEPS = 4
    NUM_EPOCHS = 1
    LEARNING_RATE = 1e-4
    MAX_LENGTH = 1024
    
    SYSTEM_PROMPT = "你是一个医学专家。请分析医学问题并给出正确答案，回答需包含思考过程。"
    OUTPUT_BASE_DIR = "./autodl-tmp/output/Qwen3-1.7B"

# ==================== 2. 数据格式化 (优化题型识别) ====================
def format_cmb_data(input_path, output_path):
    formatted_data = []
    if not os.path.exists(input_path):
        print(f"警告：找不到文件 {input_path}")
        return None
        
    with open(input_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    for item in raw_data:
        options = item.get("option", {})
        opt_str = "\n".join([f"{k}. {v}" for k, v in options.items() if v])
        
        # 提取元数据增强指令
        q_type = item.get("question_type", "单项选择题")
        subject = item.get("exam_subject", "医学基础")
        
        formatted_data.append({
            "instruction": TrainingConfig.SYSTEM_PROMPT,
            "input": f"【{subject} | {q_type}】\n问题：{item['question']}\n选项：\n{opt_str}",
            "output": f"<think>\n此题属于{subject}范畴，题型为{q_type}。\n分析：经过医学逻辑推导，该题的考点在于患者的临床表现与影像学特征的结合。\n</think>\n答案是 {item['answer']}"
        })
        
    with open(output_path, "w", encoding="utf-8") as f:
        for ex in formatted_data:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    return output_path

# ==================== 3. 核心训练逻辑 (含 Label Masking) ====================
def run_experiment(use_dora=False):
    exp_type = "dora-sft" if use_dora else "lora-sft"
    output_dir = os.path.join(TrainingConfig.OUTPUT_BASE_DIR, exp_type)
    print(f"\n>>> 启动 {exp_type.upper()} 训练任务...")

    # 下载并加载模型
    model_dir = snapshot_download(TrainingConfig.MODEL_NAME, cache_dir=TrainingConfig.MODEL_CACHE_DIR)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_dir, 
        torch_dtype=torch.bfloat16, 
        device_map="auto", 
        trust_remote_code=True
    )

    # PEFT 配置
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=TrainingConfig.LORA_R,
        lora_alpha=TrainingConfig.LORA_ALPHA,
        target_modules=TrainingConfig.LORA_TARGET_MODULES,
        use_dora=use_dora,
        lora_dropout=0.1
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 数据加载
    train_dataset = Dataset.from_json("temp_train.jsonl")
    val_dataset = Dataset.from_json("temp_val.jsonl")

    # 精确的 Label Masking 逻辑
    def tokenize_fn(ex):
        # 1. 构建 Prompt 部分
        prompt_messages = [
            {"role": "system", "content": ex["instruction"]},
            {"role": "user", "content": ex["input"]}
        ]
        prompt_text = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
        prompt_ids = tokenizer(prompt_text, truncation=True, max_length=TrainingConfig.MAX_LENGTH)["input_ids"]
        
        # 2. 构建完整对话（Prompt + Answer）
        full_messages = prompt_messages + [{"role": "assistant", "content": ex["output"]}]
        full_text = tokenizer.apply_chat_template(full_messages, tokenize=False)
        full_ids = tokenizer(full_text, truncation=True, max_length=TrainingConfig.MAX_LENGTH)["input_ids"]
        
        # 3. 构造 Labels：Prompt 部分填充 -100
        labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
        
        # 4. 长度对齐
        input_ids = full_ids[:TrainingConfig.MAX_LENGTH]
        labels = labels[:TrainingConfig.MAX_LENGTH]
        
        return {"input_ids": input_ids, "labels": labels}

    train_ds = train_dataset.map(tokenize_fn, remove_columns=train_dataset.column_names)
    val_ds = val_dataset.map(tokenize_fn, remove_columns=val_dataset.column_names)

    # 训练参数
    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=TrainingConfig.BATCH_SIZE,
        gradient_accumulation_steps=TrainingConfig.GRADIENT_ACCUMULATION_STEPS,
        num_train_epochs=TrainingConfig.NUM_EPOCHS,
        learning_rate=TrainingConfig.LEARNING_RATE,
        lr_scheduler_type="cosine",
        bf16=True,
        logging_steps=5,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        report_to="swanlab",
        remove_unused_columns=False
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True)
    )

    trainer.train()
    
    # 保存结果
    final_save_path = os.path.join(output_dir, "final_adapter")
    model.save_pretrained(final_save_path)
    tokenizer.save_pretrained(final_save_path)
    print(f"{exp_type.upper()} 训练完成。")
    
    # 显存回收
    del model, trainer
    torch.cuda.empty_cache()

# ==================== 4. 主入口 ====================
if __name__ == "__main__":
    # 初始化数据
    print("正在预处理数据...")
    format_cmb_data(TrainingConfig.TRAIN_JSON_PATH, "temp_train.jsonl")
    format_cmb_data(TrainingConfig.VAL_JSON_PATH, "temp_val.jsonl")
    
    # 启动 SwanLab 实验追踪（可选）
    swanlab.init(project="CMB-Medical-Qwen3", experiment_name="Qwen3-1.7B-Medical-LoRA")
    
    # 运行实验
    run_experiment(use_dora=False) # 建议小模型开启 DoRA 获得更强性能