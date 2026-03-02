import os
import torch
from transformers import TrainingArguments, Trainer, DataCollatorForSeq2Seq
from config.training_config import TrainingConfig

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