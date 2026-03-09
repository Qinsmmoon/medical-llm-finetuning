import torch
import json
import re
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import os
import time  # 添加时间统计

# ==================== 配置 ====================
BASE_PATH = "./autodl-tmp/Qwen/Qwen3-1.7B"
LORA_PATH = "./autodl-tmp/output/Qwen3-1.7B/lora-ft/final_adapter"
DORA_PATH = "./autodl-tmp/output/Qwen3-1.7B/dora-ft/final_adapter"
TEST_DATA = "data/CMB/CMB-Exam/CMB-val/CMB-val-merge.json"

class MedicalEvaluator:
    def __init__(self, base_path):
        self.tokenizer = AutoTokenizer.from_pretrained(base_path, trust_remote_code=True)
        self.base_path = base_path

    def extract_answer(self, text, is_multi=False):
        # 移除思考过程
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        # 预处理：去掉空格和特殊字符
        text = text.strip()

        # 1. 优先匹配：强格式 [[ABC]]
        match = re.search(r'\[\[([A-E,]+)\]\]', text, re.IGNORECASE)
        if match:
            found = re.findall(r'[A-E]', match.group(1).upper())
            return "".join(sorted(list(set(found))))

        # 2. 次优先：匹配“答案是 X”、“选项是 X”
        # 这种模式能抓取“答案是 C”、“选择C”、“答案为A,B”等
        patterns = [
            r'[答选][案项][是为][:：\s]*([A-E,]+)',
            r'the answer is[:\s]*([A-E,]+)',
            r'final answer[:\s]*([A-E,]+)'
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                found = re.findall(r'[A-E]', match.group(1).upper())
                if found:
                    return "".join(sorted(list(set(found))))

        # 3. 最后兜底：从文本末尾向前寻找 A-E 字符
        # 很多模型喜欢在最后说“所以选C”
        if is_multi:
            # 多选：找最后一段连续的 A-E
            # 比如 "所以答案是 BCD。" -> 匹配到 BCD
            matches = re.findall(r'[A-E]', text.upper())
            if matches:
                # 这种方法有风险，通常取最后几个
                return "".join(sorted(list(set(matches[-3:])))) # 假设多选最多3-4个
        else:
            # 单选：找最后一个出现的 A-E
            match = re.findall(r'[A-E]', text.upper())
            if match:
                return match[-1]

        return "None"
    

    def evaluate_model(self, adapter_path, name):
        model = AutoModelForCausalLM.from_pretrained(
            self.base_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
        )
        if adapter_path:
            model = PeftModel.from_pretrained(model, adapter_path)
        model.eval()

        with open(TEST_DATA, 'r', encoding='utf-8') as f:
            dataset = json.load(f)

        # 只取前10条数据测试
        # print(f"\n{'='*60}")
        # print(f"开始测试 {name} 模型的前 {len(dataset)} 条数据")
        # print(f"{'='*60}")

        results = []
        total_time = 0
        
        for idx, item in enumerate(tqdm(dataset, desc=f"评估 {name}")):
            start_time = time.time()
            
            is_multi = "多项" in item.get('question_type', '')

            # print(f"\n{'-'*40}")
            # print(f"【第 {idx+1} 条数据】")
            # print(f"考试类型：{item.get('exam_type', '未知')}")
            # print(f"考试科目：{item.get('exam_subject', '未知')}")
            # print(f"问题类型：{item.get('question_type', '未知')}")
            # print(f"问题：{item['question']}")

            # 处理选项 - 根据您的数据结构，选项在 "option" 键中
            if 'option' in item:
                options = item['option']
                # print("选项：")
                # for opt_key, opt_value in options.items():
                #     print(f"  {opt_key}. {opt_value}")
            else:
                print("选项：无选项")

            # 修复：构建包含问题和选项的完整提示词
            if is_multi:
                type_hint = "【多选题】本题可能有多个正确选项"
                # 【修改点】增加特殊标记，如 [[答案]]
                answer_format = "请在回答的最后一行，严格使用双中括号包裹最终选项，格式如：[[ABD]]。"
            else:
                type_hint = "【单选题】本题只有一个正确选项"
                # 【修改点】增加特殊标记
                answer_format = "请在回答的最后一行，严格使用双中括号包裹最终选项，格式如：[[A]]。"
            
            # 构建包含完整信息的提示词
            if 'option' in item:
                options_text = "\n".join([f"{k}. {v}" for k, v in item['option'].items()])
                prompt = f"{type_hint}\n\n问题：{item['question']}\n\n选项：\n{options_text}\n\n{answer_format}"
            else:
                prompt = f"{type_hint}\n\n问题：{item['question']}\n\n{answer_format}"

            # print(f"\n发送的提示词：{prompt}")

            inputs = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
            )

            model_in = self.tokenizer(inputs, return_tensors="pt").to(model.device)

            with torch.no_grad():
                # 优化：使用更合理的max_new_tokens值
                out = model.generate(
                    **model_in, 
                    max_new_tokens=4096,  # 从2048降低到256，足够包含思考过程和答案
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )

            resp = self.tokenizer.decode(out[0][model_in.input_ids.shape[1]:], skip_special_tokens=True)
            display_resp = re.sub(r'<think>.*?</think>', '', resp, flags=re.DOTALL).strip()
            if not display_resp:  # 如果移除后为空，则显示原始输出
                print("移除后为空")

            pred = self.extract_answer(resp, is_multi)
            gt = str(item['answer']).upper()
            
            elapsed_time = time.time() - start_time
            total_time += elapsed_time

            # print(f"\n【模型推理结果】")
            # print(f"推理时间：{elapsed_time:.2f}秒")
            # print(f"模型输出（已移除思考过程）：{display_resp}")
            # print(f"提取的答案：{pred}")
            # print(f"正确答案：{gt}")
            # print(f"结果：{'✅ 正确' if pred == gt else '❌ 错误'}")

            if pred != gt:
                print(f"\n{'!'*20} [回答错误] {'!'*20}")
                print(f"题目索引: {idx}")
                print(f"问题原文: {item['question'][:150]}...") # 打印前150字
                print(f"\n--- 模型输出 (已移除思考过程) ---")
                print(display_resp if display_resp else "[模型未输出有效内容]")
                print(f"\n--- 结果对比 ---")
                print(f"推理答案: [{pred}]")
                print(f"正确答案: [{gt}]")
                print(f"{'!'*50}\n")
                
            results.append({
                "category": item.get('exam_subject', '其他'),
                "correct": pred == gt,
                "question": item['question'][:100] + "...",
                "prediction": pred,
                "ground_truth": gt,
                "full_response": resp,
                "inference_time": elapsed_time
            })

        # 打印汇总结果
        print(f"\n{'='*60}")
        print(f"测试完成！前{len(dataset)}条数据结果汇总：")
        correct_count = sum(1 for r in results if r['correct'])
        print(f"正确数：{correct_count}/{len(dataset)}，准确率：{correct_count/len(dataset)*100:.1f}%")
        print(f"平均推理时间：{total_time/len(dataset):.2f}秒/条")
        print(f"总推理时间：{total_time:.2f}秒")
        print(f"{'='*60}")

        del model
        torch.cuda.empty_cache()
        return results


def main():
    evaluator = MedicalEvaluator(BASE_PATH)
    #configs = [("Base", None)]
    configs = [("DoRA", DORA_PATH)]
    summary = []
    
    for name, path in configs:
        if path and not os.path.exists(path) and name != "Base":
            continue
        res = evaluator.evaluate_model(path, name)
        
        # 按学科汇总
        df = pd.DataFrame(res)
        overall_acc = df['correct'].mean() * 100
        summary.append({"Model": name, "Category": "Overall", "Accuracy": f"{overall_acc:.2f}%"})
        
        cat_acc = df.groupby('category')['correct'].mean() * 100
        for cat, acc in cat_acc.items():
            summary.append({"Model": name, "Category": cat, "Accuracy": f"{acc:.2f}%"})

    if summary:
        report = pd.DataFrame(summary).pivot(index='Category', columns='Model', values='Accuracy')
        print("\n" + report.to_markdown())
        report.to_csv("b3.csv")
        print("\n结果已保存到 b3.csv")

if __name__ == "__main__":
    main()