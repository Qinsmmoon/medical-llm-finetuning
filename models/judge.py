import torch
import re
import json
from typing import List, Dict, Optional, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from utils.logger import setup_logger, LoggerMixin
from config.training_config import TrainingConfig

class JudgeEvaluator(LoggerMixin):
    """LLM-as-a-Judge 评估器类"""
    
    def __init__(self, judge_model_name: str = "Qwen/Qwen2.5-7B-Instruct", use_quantization: bool = True):
        """
        初始化裁判评估器
        
        Args:
            judge_model_name: 裁判模型名称
            use_quantization: 是否使用4-bit量化（节省显存）
        """
        self.judge_model_name = judge_model_name
        self.use_quantization = use_quantization
        self._logger = setup_logger(self.__class__.__name__)
        self._load_judge_model()
    
    def _load_judge_model(self):
        """加载裁判模型"""
        self.logger.info(f"加载裁判模型: {self.judge_model_name}")
        
        try:
            # 量化配置
            bnb_config = None
            if self.use_quantization:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
            
            # 加载分词器
            self.judge_tokenizer = AutoTokenizer.from_pretrained(
                self.judge_model_name,
                trust_remote_code=True
            )
            
            # 加载模型
            self.judge_model = AutoModelForCausalLM.from_pretrained(
                self.judge_model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
            )
            
            self.judge_model.eval()
            self.logger.info("裁判模型加载成功")
            
        except Exception as e:
            self.logger.error(f"裁判模型加载失败: {str(e)}")
            raise
    
    def _extract_score(self, response: str, line_prefix: str) -> Optional[float]:
        """
        从裁判回复中提取分数
        
        Args:
            response: 裁判模型的回复
            line_prefix: 分数行的前缀（如"基座模型分数"）
        
        Returns:
            Optional[float]: 提取的分数，如果没有找到则返回None
        """
        pattern = rf"{line_prefix}:\s*(\d+\.?\d*)"
        match = re.search(pattern, response)
        return float(match.group(1)) if match else None
    
    def _create_evaluation_prompt(self, question: str, base_response: str, 
                                  lora_response: str, dora_response: str) -> str:
        """
        创建评估提示词
        
        Args:
            question: 问题
            base_response: 基座模型的回答
            lora_response: LoRA模型的回答
            dora_response: DoRA模型的回答
        
        Returns:
            str: 评估提示词
        """
        prompt = f"""你是一个公正的评分员。请对以下三个针对同一问题的回答进行评分，评分维度为：相关性、准确性、完整性。
每个维度满分5分，请给出三个回答各自的总分（三个维度的平均分，保留一位小数）。
输出格式要求：每个回答的分数单独一行，例如：
基座模型分数: 4.2
LoRA模型分数: 4.5
DoRA模型分数: 4.3

问题：{question}

回答1（基座模型）：
{base_response}

回答2（LoRA微调模型）：
{lora_response}

回答3（DoRA微调模型）：
{dora_response}

请评分："""
        return prompt
    
    def evaluate_single(self, question: str, base_response: str, 
                        lora_response: str, dora_response: str) -> Dict:
        """
        评估单个问题的回答
        
        Returns:
            Dict: 包含各模型分数的字典
        """
        prompt = self._create_evaluation_prompt(
            question, base_response, lora_response, dora_response
        )
        
        messages = [{"role": "user", "content": prompt}]
        text = self.judge_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = self.judge_tokenizer(text, return_tensors="pt").to(self.judge_model.device)
        
        with torch.no_grad():
            outputs = self.judge_model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.2,
                do_sample=False
            )
        
        response = self.judge_tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 提取分数
        base_score = self._extract_score(response, "基座模型分数")
        lora_score = self._extract_score(response, "LoRA模型分数")
        dora_score = self._extract_score(response, "DoRA模型分数")
        
        return {
            "base": base_score,
            "lora": lora_score,
            "dora": dora_score,
            "full_response": response
        }
    
    def evaluate(self, questions: List[str], base_responses: List[str],
                 lora_responses: List[str], dora_responses: List[str]) -> Dict:
        """
        批量评估多个问题的回答
        
        Returns:
            Dict: 包含平均分、最佳计数和所有分数的字典
        """
        self.logger.info(f"开始评估 {len(questions)} 个问题")
        
        scores = {"base": [], "lora": [], "dora": []}
        best_counts = {"base": 0, "lora": 0, "dora": 0}
        detailed_results = []
        
        for i, (question, base_resp, lora_resp, dora_resp) in enumerate(zip(
            questions, base_responses, lora_responses, dora_responses
        )):
            self.logger.info(f"评估问题 {i+1}/{len(questions)}")
            
            result = self.evaluate_single(question, base_resp, lora_resp, dora_resp)
            
            # 收集分数
            if result["base"] is not None:
                scores["base"].append(result["base"])
            if result["lora"] is not None:
                scores["lora"].append(result["lora"])
            if result["dora"] is not None:
                scores["dora"].append(result["dora"])
            
            # 确定最佳回答
            valid_scores = [
                (name, result[name]) 
                for name in ["base", "lora", "dora"] 
                if result[name] is not None
            ]
            
            if valid_scores:
                best = max(valid_scores, key=lambda x: x[1])[0]
                best_counts[best] += 1
            
            detailed_results.append({
                "question": question,
                "scores": {k: result[k] for k in ["base", "lora", "dora"]},
                "judge_response": result["full_response"]
            })
            
            self.logger.info(f"问题 {i+1} 评分: base={result['base']}, "
                           f"lora={result['lora']}, dora={result['dora']}")
        
        # 计算平均分
        avg_scores = {
            k: sum(v)/len(v) if v else 0 
            for k, v in scores.items()
        }
        
        self.logger.info("评估完成")
        
        return {
            "avg_scores": avg_scores,
            "best_counts": best_counts,
            "all_scores": scores,
            "detailed_results": detailed_results
        }
    
    def print_results(self, results: Dict):
        """打印评估结果"""
        print("\n" + "=" * 60)
        print("LLM-as-a-Judge 对比结果")
        print("平均分:")
        for model, score in results["avg_scores"].items():
            model_name = {
                "base": "基座模型",
                "lora": "LoRA模型",
                "dora": "DoRA模型"
            }.get(model, model)
            print(f"  {model_name}: {score:.2f}")
        
        print("\n最佳回答次数:")
        for model, count in results["best_counts"].items():
            model_name = {
                "base": "基座模型",
                "lora": "LoRA模型",
                "dora": "DoRA模型"
            }.get(model, model)
            print(f"  {model_name}: {count}")
        print("=" * 60)
    
    def save_results(self, results: Dict, questions: List[str], 
                     base_responses: List[str], lora_responses: List[str],
                     dora_responses: List[str], save_path: str):
        """
        保存评估结果到文件
        
        Args:
            results: 评估结果字典
            questions: 问题列表
            base_responses: 基座模型回答列表
            lora_responses: LoRA模型回答列表
            dora_responses: DoRA模型回答列表
            save_path: 保存路径
        """
        save_data = {
            "test_questions": questions,
            "base_responses": base_responses,
            "lora_responses": lora_responses,
            "dora_responses": dora_responses,
            "judge_scores": results["all_scores"],
            "avg_scores": results["avg_scores"],
            "best_counts": results["best_counts"],
            "detailed_results": results.get("detailed_results", [])
        }
        
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"对比结果已保存到: {save_path}")
    
    def __del__(self):
        """析构函数，释放模型显存"""
        if hasattr(self, 'judge_model'):
            del self.judge_model
            torch.cuda.empty_cache()
            self.logger.info("裁判模型已释放")