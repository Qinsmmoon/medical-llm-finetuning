import torch
from modelscope import snapshot_download
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from config.training_config import TrainingConfig

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

    # 量化配置
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

    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        use_cache=False,
    )

    # 量化后处理
    if TrainingConfig.USE_4BIT_QUANTIZATION:
        model = prepare_model_for_kbit_training(model)

    # 配置LoRA/DoRA
    print("配置 LoRA/DoRA 适配器...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=TrainingConfig.LORA_R,
        lora_alpha=TrainingConfig.LORA_ALPHA,
        lora_dropout=TrainingConfig.LORA_DROPOUT,
        target_modules=TrainingConfig.LORA_TARGET_MODULES,
        bias="none",
        modules_to_save=None,
        use_dora=use_dora,
    )

    model = get_peft_model(model, lora_config)
    model.enable_input_require_grads()
    
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    model.print_trainable_parameters()
    print("模型和分词器加载完成！")
    print("=" * 50)

    return model, tokenizer