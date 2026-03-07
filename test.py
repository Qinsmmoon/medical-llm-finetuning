import json
import pandas as pd
import torch
import os
import re
from datasets import Dataset
from modelscope import snapshot_download, AutoTokenizer
from transformers import (
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig,
    pipeline
)
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    PeftModel,
    prepare_model_for_kbit_training
)
import swanlab
from typing import List, Dict, Optional

# ==================== 配置区域 ====================
class TrainingConfig:
    """训练配置类，集中管理所有超参数"""

    # 模型配置
    MODEL_NAME = "Qwen/Qwen3-1.7B"
    MODEL_CACHE_DIR = "./autodl-tmp/"

    # LoRA/DoRA 配置
    LORA_R = 8
    LORA_ALPHA = 32
    LORA_DROPOUT = 0.1
    LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]
    USE_DORA = False  # True 表示使用 DoRA，False 表示使用 LoRA

    # 训练参数
    BATCH_SIZE = 2
    GRADIENT_ACCUMULATION_STEPS = 8
    NUM_EPOCHS = 3
    LEARNING_RATE = 2e-4
    MAX_LENGTH = 2048

    # 提示词模板
    SYSTEM_PROMPT = "你是一个医学专家，你需要根据用户的问题，给出带有思考的回答。"

    # 路径配置
    OUTPUT_BASE_DIR = "./autodl-tmp/output/Qwen3-1.7B"
    TRAIN_DATA_PATH = "train.jsonl"
    VAL_DATA_PATH = "val.jsonl"

    # 是否使用4-bit量化（QLoRA）
    USE_4BIT_QUANTIZATION = False

    # SwanLab 配置
    SWANLAB_PROJECT = "qwen3-sft-medical-comparison"
    SWANLAB_RUN_NAME = "qwen3-1.7B-lora-dora-experiment"

# ==================== 数据处理函数 ====================
def format_dataset_jsonl(input_path: str, output_path: str) -> None:
    """将原始JSONL数据集格式化为标准指令微调格式"""
    formatted_data = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line)
                formatted_example = {
                    "instruction": TrainingConfig.SYSTEM_PROMPT,
                    "input": data.get("question", ""),
                    "output": f"<think>{data.get('think', '')}</think>\n{data.get('answer', '')}"
                }
                formatted_data.append(formatted_example)
            except json.JSONDecodeError as e:
                print(f"JSON解析错误（行被跳过）: {e}")
                continue
    with open(output_path, "w", encoding="utf-8") as f:
        for example in formatted_data:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")
    print(f"数据集已格式化并保存到: {output_path}，共 {len(formatted_data)} 条样本")

def preprocess_function(example: Dict, tokenizer) -> Dict:
    """预处理函数：将文本数据转换为模型训练所需的token IDs"""
    conversation = [
        {"role": "system", "content": TrainingConfig.SYSTEM_PROMPT},
        {"role": "user", "content": example["input"]},
        {"role": "assistant", "content": example["output"]}
    ]
    text = tokenizer.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=False
    )
    tokenized = tokenizer(
        text,
        truncation=True,
        max_length=TrainingConfig.MAX_LENGTH,
        padding=False
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

# ==================== 模型加载与 LoRA/DoRA 配置 ====================
def load_model_and_tokenizer(use_dora: bool = False):
    """
    加载模型和分词器，并根据 use_dora 配置 LoRA 或 DoRA
    """
    print("=" * 50)
    print(f"开始加载模型和分词器，DoRA模式: {use_dora}")

    model_dir = snapshot_download(
        TrainingConfig.MODEL_NAME,
        cache_dir=TrainingConfig.MODEL_CACHE_DIR,
        revision="master"
    )
    print(f"模型位置: {model_dir}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        use_fast=False,
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    bnb_config = None
    torch_dtype = torch.bfloat16

    if TrainingConfig.USE_4BIT_QUANTIZATION:
        print("使用4-bit量化（QLoRA）...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        torch_dtype = torch.float16

    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        use_cache=False,
    )

    if TrainingConfig.USE_4BIT_QUANTIZATION:
        model = prepare_model_for_kbit_training(model)

    print("配置 LoRA/DoRA 适配器...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=TrainingConfig.LORA_R,
        lora_alpha=TrainingConfig.LORA_ALPHA,
        lora_dropout=TrainingConfig.LORA_DROPOUT,
        target_modules=TrainingConfig.LORA_TARGET_MODULES,
        bias="none",
        modules_to_save=None,
        use_dora=use_dora,  # 关键参数：True 表示 DoRA，False 表示 LoRA
    )

    model = get_peft_model(model, lora_config)

    model.enable_input_require_grads()
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    model.print_trainable_parameters()
    print("模型和分词器加载完成！")
    print("=" * 50)

    return model, tokenizer

def train_model(model, tokenizer, train_dataset, eval_dataset, output_dir: str):
    """
    训练 LoRA/DoRA 适配器，保存到指定 output_dir
    """
    print("开始配置训练参数...")

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=TrainingConfig.BATCH_SIZE,
        per_device_eval_batch_size=TrainingConfig.BATCH_SIZE,
        gradient_accumulation_steps=TrainingConfig.GRADIENT_ACCUMULATION_STEPS,
        num_train_epochs=TrainingConfig.NUM_EPOCHS,
        learning_rate=TrainingConfig.LEARNING_RATE,
        optim="adamw_8bit" if TrainingConfig.USE_4BIT_QUANTIZATION else "adamw_torch",
        weight_decay=0.01,
        warmup_ratio=0.03,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=400,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        logging_strategy="steps",
        logging_steps=10,
        report_to="swanlab",
        run_name=f"{TrainingConfig.SWANLAB_RUN_NAME}_{'dora' if model.peft_config['default'].use_dora else 'lora'}",
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        gradient_checkpointing=True,
        dataloader_num_workers=4,
        remove_unused_columns=True,
        group_by_length=True,
        dataloader_pin_memory=True,
        logging_first_step=True,
        greater_is_better=False,
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        pad_to_multiple_of=8,
        return_tensors="pt"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    print("开始训练...")
    trainer.train()

    # 保存最终适配器
    final_adapter_dir = os.path.join(output_dir, "final_adapter")
    model.save_pretrained(final_adapter_dir)
    tokenizer.save_pretrained(final_adapter_dir)
    trainer.save_model(output_dir)
    trainer.save_state()

    print(f"训练完成！模型保存在: {output_dir}")
    return trainer

# ==================== 推理类 ====================
class QwenInference:
    """支持基座模型和 LoRA/DoRA 微调模型的推理类"""

    def __init__(self, base_model_name: str = None, lora_adapter_path: str = None):
        """
        lora_adapter_path 为 None 时只加载基座模型
        """
        if base_model_name is None:
            base_model_name = './autodl-tmp/Qwen/Qwen3-1.7B'

        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_name,
            use_fast=False,
            trust_remote_code=True
        )
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )

        if lora_adapter_path is not None and os.path.exists(lora_adapter_path):
            self.model = PeftModel.from_pretrained(
                self.base_model,
                lora_adapter_path,
                device_map="auto"
            )
            self.use_lora = True
        else:
            self.model = self.base_model
            self.use_lora = False

        self.model.eval()
        print(f"推理模型加载完成，使用LoRA/DoRA: {self.use_lora}")

    def generate_response(self, question: str, max_new_tokens: int = 1324,
                          temperature: float = 0.7, do_sample: bool = True) -> str:
        messages = [
            {"role": "system", "content": TrainingConfig.SYSTEM_PROMPT},
            {"role": "user", "content": question}
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        generation_config = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "do_sample": do_sample,
            "top_p": 0.9,
            "repetition_penalty": 1.1,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
        }

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **generation_config)

        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        assistant_marker = "<|im_start|>assistant\n"
        if assistant_marker in full_response:
            response = full_response.split(assistant_marker)[-1]
        else:
            response = full_response
        return response.strip()

# ==================== LLM-as-a-Judge 评估函数 ====================
def judge_evaluation(questions: List[str],
                     base_responses: List[str],
                     lora_responses: List[str],
                     dora_responses: List[str],
                     judge_model_name: str = "Qwen/Qwen2.5-7B-Instruct") -> Dict:
    """
    使用裁判模型对三个模型的回答进行评分（1-5分，维度：相关性、准确性、完整性）
    返回每个模型的平均分和每个问题的最佳模型统计
    """
    print("加载裁判模型...")
    # 使用4-bit量化加载裁判模型以节省显存
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    judge_tokenizer = AutoTokenizer.from_pretrained(
        judge_model_name,
        trust_remote_code=True
    )
    judge_model = AutoModelForCausalLM.from_pretrained(
        judge_model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    judge_model.eval()

    scores = {"base": [], "lora": [], "dora": []}
    best_counts = {"base": 0, "lora": 0, "dora": 0}

    for i, question in enumerate(questions):
        prompt = f"""你是一个公正的评分员。请对以下三个针对同一问题的回答进行评分，评分维度为：相关性、准确性、完整性。
每个维度满分5分，请给出三个回答各自的总分（三个维度的平均分，保留一位小数）。
输出格式要求：每个回答的分数单独一行，例如：
基座模型分数: 4.2
LoRA模型分数: 4.5
DoRA模型分数: 4.3

问题：{question}

回答1（基座模型）：
{base_responses[i]}

回答2（LoRA微调模型）：
{lora_responses[i]}

回答3（DoRA微调模型）：
{dora_responses[i]}

请评分："""
        messages = [{"role": "user", "content": prompt}]
        text = judge_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = judge_tokenizer(text, return_tensors="pt").to(judge_model.device)

        with torch.no_grad():
            outputs = judge_model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.2,
                do_sample=False
            )
        response = judge_tokenizer.decode(outputs[0], skip_special_tokens=True)
        # 提取评分
        # 简单解析：查找 "基座模型分数: X.X" 等模式
        def extract_score(line_prefix):
            pattern = rf"{line_prefix}:\s*(\d+\.?\d*)"
            match = re.search(pattern, response)
            return float(match.group(1)) if match else None

        base_score = extract_score("基座模型分数")
        lora_score = extract_score("LoRA模型分数")
        dora_score = extract_score("DoRA模型分数")

        if base_score is not None:
            scores["base"].append(base_score)
        if lora_score is not None:
            scores["lora"].append(lora_score)
        if dora_score is not None:
            scores["dora"].append(dora_score)

        # 确定最佳
        valid_scores = [(name, s) for name, s in zip(["base", "lora", "dora"],
                                                      [base_score, lora_score, dora_score]) if s is not None]
        if valid_scores:
            best = max(valid_scores, key=lambda x: x[1])[0]
            best_counts[best] += 1

        print(f"问题 {i+1} 评分完成: base={base_score}, lora={lora_score}, dora={dora_score}")

    # 释放裁判模型显存
    del judge_model
    torch.cuda.empty_cache()

    # 计算平均分
    avg_scores = {k: sum(v)/len(v) if v else 0 for k, v in scores.items()}
    return {
        "avg_scores": avg_scores,
        "best_counts": best_counts,
        "all_scores": scores
    }

# ==================== 主执行流程 ====================
def main():
    print("=" * 60)
    print("Qwen3-1.7B LoRA/DoRA 微调及 LLM-as-a-Judge 对比")
    print(f"设备: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print("=" * 60)

    # 准备数据集
    print("\n[步骤1] 准备数据集...")
    train_formatted_path = "train_formatted.jsonl"
    val_formatted_path = "val_formatted.jsonl"
    if not os.path.exists(train_formatted_path):
        format_dataset_jsonl(TrainingConfig.TRAIN_DATA_PATH, train_formatted_path)
    if not os.path.exists(val_formatted_path):
        format_dataset_jsonl(TrainingConfig.VAL_DATA_PATH, val_formatted_path)

    # 加载数据集
    train_df = pd.read_json(train_formatted_path, lines=True)
    val_df = pd.read_json(val_formatted_path, lines=True)
    train_ds = Dataset.from_pandas(train_df)
    val_ds = Dataset.from_pandas(val_df)

    # 预分词（使用同一个tokenizer，因为训练时要加载模型，我们先用一个临时tokenizer做预处理）
    # 先加载一次分词器用于预处理
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

    # -------------------- 训练 LoRA 模型 --------------------
    print("\n[步骤2] 训练 LoRA 模型...")
    lora_output_dir = os.path.join(TrainingConfig.OUTPUT_BASE_DIR, "lora")
    os.makedirs(lora_output_dir, exist_ok=True)
    # 重新加载模型（LoRA模式）
    model_lora, tokenizer_lora = load_model_and_tokenizer(use_dora=False)
    train_model(model_lora, tokenizer_lora, train_dataset, eval_dataset, lora_output_dir)
    del model_lora, tokenizer_lora
    torch.cuda.empty_cache()

    # -------------------- 训练 DoRA 模型 --------------------
    print("\n[步骤3] 训练 DoRA 模型...")
    dora_output_dir = os.path.join(TrainingConfig.OUTPUT_BASE_DIR, "dora")
    os.makedirs(dora_output_dir, exist_ok=True)
    model_dora, tokenizer_dora = load_model_and_tokenizer(use_dora=True)
    train_model(model_dora, tokenizer_dora, train_dataset, eval_dataset, dora_output_dir)
    del model_dora, tokenizer_dora
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main() 