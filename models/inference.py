import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from config.training_config import TrainingConfig

class QwenInference:
    """支持基座模型和 LoRA/DoRA 微调模型的推理类"""

    def __init__(self, base_model_name: str = None, lora_adapter_path: str = None):
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