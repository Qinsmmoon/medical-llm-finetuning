import torch
import json
import re
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import os

# ==================== 路径配置 ====================
BASE_MODEL_PATH = "./autodl-tmp/Qwen/Qwen3-1.7B"
LORA_ADAPTER = "./autodl-tmp/output/Qwen3-1.7B/lora/final_adapter"
DORA_ADAPTER = "./autodl-tmp/output/Qwen3-1.7B/dora/final_adapter"
VAL_DATA_PATH = "data/CMB/CMB-Exam/CMB-val/CMB-val-merge.json"

class LocalCMBEvaluator:
    def __init__(self, base_path):
        self.base_path = base_path
        self.tokenizer = AutoTokenizer.from_pretrained(base_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def load_local_data(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def get_model(self, adapter_path=None):
        model = AutoModelForCausalLM.from_pretrained(
            self.base_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        if adapter_path and os.path.exists(adapter_path):
            print(f"-> 挂载适配器: {adapter_path}")
            model = PeftModel.from_pretrained(model, adapter_path)
        model.eval()
        return model

    def extract_answer(self, text, is_multi=False):
        """
        改进的提取逻辑：
        1. 移除思考过程
        2. 如果是多选，提取所有出现的 A-E 并排序拼接
        3. 如果是单选，提取第一个出现的 A-E
        """
        # 移除 <think> 标签内容
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        clean_text = text.upper()
        
        if is_multi:
            # 找到所有 A-E 的字母并去重排序，例如输入 "选A和C" -> "AC"
            found = re.findall(r'[A-E]', clean_text)
            return "".join(sorted(list(set(found)))) if found else "N/A"
        else:
            # 单选：匹配第一个出现的独立 A-E
            match = re.search(r'([A-E])', clean_text)
            return match.group(1) if match else "N/A"

    def run_eval(self, model, dataset, model_name, debug_count=5):
        category_stats = {}
        
        # 为了节省测试时间，你可以先只测前 N 条，如果想测全部，去掉 dataset[:N]
        # test_data = dataset[:20] # 调试时可以只用前20条
        test_data = dataset 

        for i, item in enumerate(tqdm(test_data, desc=f"评估 {model_name}")):
            # 匹配你提供的 JSON 字段
            cat = item.get('exam_subject', '未知科室')
            gt = str(item.get('answer', '')).upper().strip()
            q_type = item.get('question_type', '单项选择题')
            is_multi = "多项" in q_type
            
            if cat not in category_stats:
                category_stats[cat] = {'correct': 0, 'total': 0}
            
            options = item.get('option', {})
            opt_str = "\n".join([f"{k}. {v}" for k, v in options.items() if v])
            
            # 构建针对微调模型的 Prompt
            prompt = f"任务：{q_type}\n问题：{item['question']}\n选项：\n{opt_str}\n请直接给出正确选项的字母。"
            
            inputs = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], 
                tokenize=False, 
                add_generation_prompt=True
            )
            model_inputs = self.tokenizer(inputs, return_tensors="pt").to(model.device)

            with torch.no_grad():
                # 调低 temperature 增加确定性
                outputs = model.generate(**model_inputs, max_new_tokens=64, do_sample=False)
            
            response = self.tokenizer.decode(outputs[0][model_inputs.input_ids.shape[1]:], skip_special_tokens=True)
            pred = self.extract_answer(response, is_multi=is_multi)

            # 调试输出：检查前几条数据的匹配情况
            if i < debug_count:
                print(f"\n样本 {i+1} | 类型: {q_type}")
                print(f"预测: {pred} | 真值: {gt}")
                print(f"模型原始输出: {response.strip()[:50]}...")

            category_stats[cat]['total'] += 1
            if pred == gt:
                category_stats[cat]['correct'] += 1
                
        return category_stats

def main():
    evaluator = LocalCMBEvaluator(BASE_MODEL_PATH)
    raw_data = evaluator.load_local_data(VAL_DATA_PATH)
    
    experiments = [
        {"name": "Base", "path": None},
        {"name": "LoRA", "path": LORA_ADAPTER},
        {"name": "DoRA", "path": DORA_ADAPTER},
    ]

    all_metrics = []

    for exp in experiments:
        print(f"\n" + "="*20 + f" 正在测试: {exp['name']} " + "="*20)
        try:
            model = evaluator.get_model(exp['path'])
            stats = evaluator.run_eval(model, raw_data, exp['name'])
            
            total_correct = sum(v['correct'] for v in stats.values())
            total_q = sum(v['total'] for v in stats.values())
            
            # 记录 Overall
            all_metrics.append({
                "Model": exp['name'],
                "Category": "综合准确率 (Overall)",
                "Accuracy": f"{(total_correct/total_q)*100:.2f}%" if total_q > 0 else "0%"
            })
            
            # 记录各科室
            for cat, val in stats.items():
                acc = (val['correct']/val['total'])*100
                all_metrics.append({
                    "Model": exp['name'],
                    "Category": cat,
                    "Accuracy": f"{acc:.2f}%"
                })
                
            del model
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"测试 {exp['name']} 时出错: {e}")

    # 生成报表
    if all_metrics:
        df = pd.DataFrame(all_metrics)
        report = df.pivot(index='Category', columns='Model', values='Accuracy')
        print("\n" + "="*50)
        print("           CMB 自动化评估对比报告")
        print("="*50)
        print(report.to_markdown())
        report.to_csv("cmb_local_comparison_v2.csv")
        print("\n详细报表已保存至: cmb_local_comparison_v2.csv")

if __name__ == "__main__":
    main()