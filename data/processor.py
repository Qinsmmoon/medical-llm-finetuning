import json
from config.training_config import TrainingConfig

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
    return formatted_data

def preprocess_function(example, tokenizer):
    """预处理函数：将文本数据转换为模型训练所需的token IDs"""
    from config.training_config import TrainingConfig
    
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