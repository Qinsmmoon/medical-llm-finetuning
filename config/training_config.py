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
    USE_DORA = False
    
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
    TRAIN_DATA_PATH = "data/raw/train.jsonl"
    VAL_DATA_PATH = "data/raw/val.jsonl"
    
    # 是否使用4-bit量化
    USE_4BIT_QUANTIZATION = False
    
    # SwanLab 配置
    SWANLAB_PROJECT = "qwen3-sft-medical-comparison"
    SWANLAB_RUN_NAME = "qwen3-1.7B-lora-dora-experiment"