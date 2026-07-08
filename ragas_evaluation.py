"""
RAGAS评估模板
清晰标注问题引入部分、索引引入部分、答案引入部分
使用千问模型进行评估
"""
import os
import json
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum

try:
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness, answer_relevancy, context_precision,
        context_recall, context_relevance, answer_similarity
    )
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.run_config import RunConfig
    from datasets import Dataset
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False
    print("警告：RAGAS未安装，将使用简化评估模式")

try:
    from ragas.evaluation_schema import EvaluationDataset, SingleTurnSample
    RAGAS_NEW_API = True
except ImportError:
    RAGAS_NEW_API = False


@dataclass
class EvaluationConfig:
    """评估配置"""
    model_name: str = "qwen-turbo"
    api_key: str = ""
    embedding_model: str = "BAAI/bge-large-zh"
    chunk_size: int = 500
    chunk_overlap: int = 50
    top_k: int = 5
    evaluation_metrics: List[str] = field(default_factory=lambda: [
        "faithfulness",
        "answer_relevancy", 
        "context_precision",
        "context_recall",
        "context_relevance"
    ])
    output_directory: str = "./evaluation_results"


class EvaluationSample:
    """评估样本类"""
    
    def __init__(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: str = None,
        reference: str = None
    ):
        self.question = question
        self.answer = answer
        self.contexts = contexts
        self.ground_truth = ground_truth
        self.reference = reference
        self.timestamp = datetime.now().isoformat()
        self.id = self._generate_id()
    
    def _generate_id(self) -> str:
        """生成唯一ID"""
        import hashlib
        content = f"{self.question}{self.answer}{self.timestamp}"
        return hashlib.md5(content.encode()).hexdigest()[:8]
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "question": self.question,
            "answer": self.answer,
            "contexts": self.contexts,
            "ground_truth": self.ground_truth,
            "reference": self.reference,
            "timestamp": self.timestamp
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EvaluationSample':
        """从字典创建实例"""
        sample = cls(
            question=data["question"],
            answer=data["answer"],
            contexts=data["contexts"],
            ground_truth=data.get("ground_truth"),
            reference=data.get("reference")
        )
        if "timestamp" in data:
            sample.timestamp = data["timestamp"]
        if "id" in data:
            sample.id = data["id"]
        return sample


class EvaluationResult:
    """评估结果类"""
    
    def __init__(
        self,
        sample_id: str,
        question: str,
        answer: str,
        metrics: Dict[str, float],
        details: Dict[str, Any] = None
    ):
        self.sample_id = sample_id
        self.question = question
        self.answer = answer
        self.metrics = metrics
        self.details = details or {}
        self.timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "sample_id": self.sample_id,
            "question": self.question,
            "answer": self.answer,
            "metrics": self.metrics,
            "details": self.details,
            "timestamp": self.timestamp
        }
    
    def to_dataframe_row(self) -> Dict[str, Any]:
        """转换为DataFrame行"""
        row = {
            "sample_id": self.sample_id,
            "question": self.question,
            "answer": self.answer,
            "timestamp": self.timestamp
        }
        row.update(self.metrics)
        return row


class RAGASEvaluator:
    """RAGAS评估器"""
    
    def __init__(self, config: EvaluationConfig = None):
        self.config = config or EvaluationConfig()
        self.llm_model = None
        self.embeddings = None
        self._init_models()
    
    def _init_models(self):
        """初始化评估模型"""
        if not RAGAS_AVAILABLE:
            return
        
        try:
            from langchain_community.llms import Tongyi
            self.llm_model = Tongyi(model="qwen-turbo")
            
            from langchain_huggingface import HuggingFaceEmbeddings
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.config.embedding_model
            )
        except Exception as e:
            print(f"模型初始化警告：{e}")
            self.llm_model = None
            self.embeddings = None
    
    def _get_ragas_metrics(self):
        """获取RAGAS评估指标"""
        if not RAGAS_AVAILABLE:
            return {}
        
        metrics = {}
        wrapped_llm = LangchainLLMWrapper(self.llm_model)
        wrapped_embeddings = LangchainEmbeddingsWrapper(self.embeddings)
        
        metric_configs = {
            "faithfulness": faithfulness,
            "answer_relevancy": answer_relevancy,
            "context_precision": context_precision,
            "context_recall": context_recall,
            "context_relevance": context_relevance
        }
        
        for metric_name in self.config.evaluation_metrics:
            if metric_name in metric_configs:
                try:
                    if metric_name in ["answer_relevancy", "context_relevance"]:
                        metrics[metric_name] = metric_configs[metric_name](
                            llm=wrapped_llm,
                            embeddings=wrapped_embeddings
                        )
                    else:
                        metrics[metric_name] = metric_configs[metric_name](
                            llm=wrapped_llm
                        )
                except Exception as e:
                    print(f"指标 {metric_name} 初始化失败：{e}")
        
        return metrics
    
    def evaluate_sample(self, sample: EvaluationSample) -> EvaluationResult:
        """
        评估单个样本
        
        Args:
            sample: 评估样本
        
        Returns:
            评估结果
        """
        if not RAGAS_AVAILABLE:
            return self._simplified_evaluate(sample)
        
        try:
            from datasets import Dataset
            
            data = {
                "question": [sample.question],
                "answer": [sample.answer],
                "contexts": [sample.contexts],
                "ground_truth": [sample.ground_truth] if sample.ground_truth else [""]
            }
            
            dataset = Dataset.from_dict(data)
            metrics = self._get_ragas_metrics()
            
            if not metrics:
                return self._simplified_evaluate(sample)
            
            results = evaluate(
                dataset=dataset,
                metrics=list(metrics.values()),
                run_config=RunConfig(timeout=300)
            )
            
            score_dict = {}
            for metric_name, metric in metrics.items():
                try:
                    score = results[metric_name].values[0]
                    score_dict[metric_name] = float(score) if score is not None else 0.0
                except Exception:
                    score_dict[metric_name] = 0.0
            
            return EvaluationResult(
                sample_id=sample.id,
                question=sample.question,
                answer=sample.answer,
                metrics=score_dict,
                details={"full_results": str(results)}
            )
            
        except Exception as e:
            print(f"评估过程出错：{e}")
            return self._simplified_evaluate(sample)
    
    def _simplified_evaluate(self, sample: EvaluationSample) -> EvaluationResult:
        """
        简化评估方法（当RAGAS不可用时使用）
        
        Args:
            sample: 评估样本
        
        Returns:
            评估结果
        """
        metrics = {}
        
        try:
            if "faithfulness" in self.config.evaluation_metrics:
                metrics["faithfulness"] = self._calculate_faithfulness(
                    sample.question, sample.answer, sample.contexts
                )
            
            if "answer_relevancy" in self.config.evaluation_metrics:
                metrics["answer_relevancy"] = self._calculate_answer_relevancy(
                    sample.question, sample.answer
                )
            
            if "context_precision" in self.config.evaluation_metrics:
                metrics["context_precision"] = self._calculate_context_precision(
                    sample.contexts
                )
            
            if "context_recall" in self.config.evaluation_metrics:
                metrics["context_recall"] = self._calculate_context_recall(
                    sample.question, sample.contexts
                )
            
        except Exception as e:
            print(f"简化评估计算错误：{e}")
        
        return EvaluationResult(
            sample_id=sample.id,
            question=sample.question,
            answer=sample.answer,
            metrics=metrics
        )
    
    def _calculate_faithfulness(
        self, 
        question: str, 
        answer: str, 
        contexts: List[str]
    ) -> float:
        """
        计算忠实度指标
        评估回答是否忠实于检索到的上下文
        """
        if not answer or not contexts:
            return 0.0
        
        answer_lower = answer.lower()
        context_text = " ".join(contexts).lower()
        
        answer_words = set(answer_lower.split())
        context_words = set(context_text.split())
        
        if len(answer_words) == 0:
            return 1.0
        
        matches = len(answer_words & context_words)
        return min(1.0, matches / len(answer_words))
    
    def _calculate_answer_relevancy(
        self, 
        question: str, 
        answer: str
    ) -> float:
        """
        计算回答相关性指标
        评估回答与问题的相关程度
        """
        if not answer or not question:
            return 0.0
        
        question_keywords = set(question.lower().split())
        answer_words = set(answer.lower().split())
        
        if len(question_keywords) == 0:
            return 1.0
        
        matches = len(question_keywords & answer_words)
        return min(1.0, matches / len(question_keywords))
    
    def _calculate_context_precision(self, contexts: List[str]) -> float:
        """
        计算上下文精确度指标
        评估检索到的上下文的精确程度
        """
        if not contexts:
            return 0.0
        
        non_empty = sum(1 for ctx in contexts if ctx.strip())
        return min(1.0, non_empty / len(contexts))
    
    def _calculate_context_recall(
        self, 
        question: str, 
        contexts: List[str]
    ) -> float:
        """
        计算上下文召回率指标
        评估检索到的上下文是否覆盖问题要点
        """
        if not contexts or not question:
            return 0.0
        
        question_words = set(question.lower().split())
        context_text = " ".join(contexts).lower()
        
        matches = sum(1 for word in question_words if word in context_text)
        return min(1.0, matches / len(question_words)) if question_words else 1.0
    
    def evaluate_dataset(
        self, 
        samples: List[EvaluationSample],
        show_progress: bool = True
    ) -> List[EvaluationResult]:
        """
        评估整个数据集
        
        Args:
            samples: 评估样本列表
            show_progress: 是否显示进度
        
        Returns:
            评估结果列表
        """
        results = []
        total = len(samples)
        
        for i, sample in enumerate(samples):
            if show_progress:
                print(f"评估进度：{i+1}/{total}")
            
            result = self.evaluate_sample(sample)
            results.append(result)
        
        return results
    
    def generate_report(
        self, 
        results: List[EvaluationResult],
        output_path: str = None
    ) -> Dict[str, Any]:
        """
        生成评估报告
        
        Args:
            results: 评估结果列表
            output_path: 输出路径
        
        Returns:
            报告字典
        """
        if not results:
            return {}
        
        df = pd.DataFrame([r.to_dataframe_row() for r in results])
        
        metric_columns = [col for col in df.columns if col not in 
                         ["sample_id", "question", "answer", "timestamp"]]
        
        report = {
            "summary": {
                "total_samples": len(results),
                "evaluation_time": datetime.now().isoformat()
            },
            "metrics_summary": {},
            "statistics": {},
            "results": [r.to_dict() for r in results]
        }
        
        for metric in metric_columns:
            if metric in df.columns:
                report["statistics"][metric] = {
                    "mean": float(df[metric].mean()),
                    "std": float(df[metric].std()),
                    "min": float(df[metric].min()),
                    "max": float(df[metric].max()),
                    "median": float(df[metric].median())
                }
                report["metrics_summary"][metric] = {
                    "average_score": float(df[metric].mean()),
                    "score_level": self._get_score_level(df[metric].mean())
                }
        
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            df.to_csv(output_path.replace(".json", "_details.csv"), 
                     index=False, encoding="utf-8-sig")
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
        
        return report
    
    def _get_score_level(self, score: float) -> str:
        """根据分数获取等级"""
        if score >= 0.9:
            return "优秀"
        elif score >= 0.7:
            return "良好"
        elif score >= 0.5:
            return "一般"
        else:
            return "需改进"


class RAGASEvaluationTemplate:
    """
    RAGAS评估模板类
    提供标准化的RAG评估流程
    """
    
    def __init__(self, config: EvaluationConfig = None):
        self.config = config or EvaluationConfig()
        self.evaluator = RAGASEvaluator(self.config)
    
    def create_sample(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: str = None
    ) -> EvaluationSample:
        """
        创建评估样本
        
        Args:
            question: 问题引入部分
            answer: 答案引入部分  
            contexts: 索引引入部分（检索到的上下文）
            ground_truth: 标准答案（可选）
        
        Returns:
            评估样本
        """
        return EvaluationSample(
            question=question,
            answer=answer,
            contexts=contexts,
            ground_truth=ground_truth
        )
    
    def run_evaluation(
        self,
        questions: List[str],
        answers: List[str],
        all_contexts: List[List[str]],
        ground_truths: List[str] = None,
        output_file: str = None
    ) -> Dict[str, Any]:
        """
        运行完整评估流程
        
        Args:
            questions: 问题列表（问题引入部分）
            answers: 答案列表（答案引入部分）
            all_contexts: 所有上下文列表（索引引入部分）
            ground_truths: 标准答案列表（可选）
        
        Returns:
            评估报告
        """
        samples = []
        for i in range(len(questions)):
            sample = self.create_sample(
                question=questions[i],
                answer=answers[i],
                contexts=all_contexts[i],
                ground_truth=ground_truths[i] if ground_truths else None
            )
            samples.append(sample)
        
        results = self.evaluator.evaluate_dataset(samples)
        
        output_path = output_file or os.path.join(
            self.config.output_directory,
            f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        
        report = self.evaluator.generate_report(results, output_path)
        
        return {
            "results": results,
            "report": report,
            "samples": samples
        }
    
    def print_sample_analysis(self, sample: EvaluationSample):
        """打印样本分析结果"""
        print("\n" + "="*60)
        print("评估样本分析")
        print("="*60)
        
        print("\n【问题引入部分】")
        print(f"问题：{sample.question}")
        
        print("\n【索引引入部分】")
        print(f"检索到的上下文数量：{len(sample.contexts)}")
        for j, ctx in enumerate(sample.contexts[:3], 1):
            preview = ctx[:200] + "..." if len(ctx) > 200 else ctx
            print(f"上下文 {j}：{preview}")
        
        print("\n【答案引入部分】")
        print(f"回答：{sample.answer}")
        
        if sample.ground_truth:
            print(f"\n标准答案：{sample.ground_truth}")


def main():
    """主函数示例"""
    config = EvaluationConfig(
        model_name="qwen-turbo",
        api_key=os.environ.get("DASHSCOPE_API_KEY", ""),
        evaluation_metrics=[
            "faithfulness",
            "answer_relevancy",
            "context_precision",
            "context_recall"
        ]
    )
    
    template = RAGASEvaluationTemplate(config)
    
    sample_questions = [
        "这个保险产品的保障范围是什么？",
        "理赔需要准备哪些材料？",
        "保费是如何计算的？"
    ]
    
    sample_answers = [
        "该保险产品主要提供人身意外伤害保障，包括意外身故、伤残、医疗等保障项目。",
        "理赔需要准备保险单原件、身份证明、医院诊断证明、费用清单等相关材料。",
        "保费根据被保险人的年龄、职业类别、保障额度等因素综合计算。"
    ]
    
    sample_contexts = [
        [
            "本保险产品保障范围包括：意外身故保险金、意外伤残保险金、意外医疗保险金。",
            "保障期限为一年，保险金额根据投保方案不同分为10万、30万、50万三档。",
            "被保险人年龄范围为18-60周岁，从事1-3类职业的人员可投保。"
        ],
        [
            "理赔所需材料清单：1. 保险合同或保险单；2. 被保险人有效身份证件；3. 医院出具的诊断证明、病历资料；4. 医疗费用发票和费用清单；5. 意外事故证明（如有）。",
            "理赔申请应在事故发生后30日内提出，超出时限可能影响理赔。",
            "保险公司收到完整材料后，10个工作日内完成审核并支付赔款。"
        ],
        [
            "保费计算公式：基础保费 × 职业类别系数 × 年龄系数 × 保障额度系数。",
            "1-3类职业系数分别为1.0、1.2、1.5，年龄系数根据年龄段有所不同。",
            "基础保费为500元/年（10万保额），具体保费以投保时系统计算为准。"
        ]
    ]
    
    print("开始RAGAS评估...")
    eval_result = template.run_evaluation(
        questions=sample_questions,
        answers=sample_answers,
        all_contexts=sample_contexts
    )
    
    print("\n评估完成！")
    print(f"评估样本数：{len(eval_result['results'])}")
    
    if eval_result['report'].get('statistics'):
        print("\n评估指标统计：")
        for metric, stats in eval_result['report']['statistics'].items():
            print(f"  {metric}：均值 {stats['mean']:.4f}，标准差 {stats['std']:.4f}")


if __name__ == "__main__":
    main()
