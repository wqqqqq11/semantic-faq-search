import json
import os
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from fastapi import HTTPException, UploadFile
from typing import List, Dict, Any, Optional

from src.utils.common import load_config, setup_logger
from src.models.models import CLIPEmbedder, TestRequest, TestMetrics, TestResponse
from src.repositories.milvus_store import MilvusStore
from src.utils.metrics import MetricsCollector


class QASTestService:
    def __init__(self):
        self.config = load_config()
        self.test_config = self._load_test_config()
        self.logger = setup_logger("QASTestService", self.config)
        self.embedder = CLIPEmbedder(self.config)
        self.store = MilvusStore(self.config)
        self.metrics_collector = MetricsCollector(self.config)
        
    def _load_test_config(self) -> Dict[str, Any]:
        """
        加载测试配置
        """
        test_config_path = "tool_configs/test_config.json"
        with open(test_config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _load_test_data(self, csv_path: str) -> pd.DataFrame:
        """
        加载测试数据
        """
        
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"测试文件不存在: {csv_path}")
            
        df = pd.read_csv(csv_path, encoding='utf-8')
            
        required_columns = [
            self.test_config['test_data']['question_column'],
            self.test_config['test_data']['answer_column'],
            self.test_config['test_data']['id_column']
        ]
            
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"测试数据缺少必要列: {missing_columns}")
            
        self.logger.info(f"成功加载测试数据，共 {len(df)} 条记录")
        return df
            
    
    def _search_vector_database(self, question: str, top_k: int) -> List[Dict[str, Any]]:
        """
        向量数据库搜索
        """
        query_vector = self.embedder.encode([question])[0]
        results = self.store.search(query_vector=query_vector, top_k=top_k)
        return results
    
    def _calculate_recall_metrics(self, test_results: List[Dict[str, Any]], 
                                recall_k_values: List[int]) -> Dict[str, float]:
        """
        计算召回率指标
        """
        metrics = {}
        total_queries = len(test_results)
        
        if total_queries == 0:
            return {f"recall_at_{k}": 0.0 for k in recall_k_values}
        
        for k in recall_k_values:
            hits = sum(1 for result in test_results if result.get(f'hit_at_{k}', False))
            recall = hits / total_queries
            metrics[f"recall_at_{k}"] = round(recall, 4)
        
        return metrics
    
    def _exact_match(self, answer1: str, answer2: str) -> bool:
        """
        精确匹配
        """
        if pd.isna(answer1) or pd.isna(answer2):
            return False
        return str(answer1).strip() == str(answer2).strip()
    
    def _process_single_query(self, row: pd.Series, top_k: int, 
                            recall_k_values: List[int]) -> Dict[str, Any]:
        """
        处理单个查询
        """
        question_col = self.test_config['test_data']['question_column']
        answer_col = self.test_config['test_data']['answer_column']
        id_col = self.test_config['test_data']['id_column']
        
        query_id = row[id_col]
        question = row[question_col]
        expected_answer = row[answer_col]
        
        search_results = self._search_vector_database(question, top_k)
        
        result = {
            'query_id': query_id,
            'question': question,
            'expected_answer': expected_answer,
            'search_results_count': len(search_results),
            'top_results': []
        }
        
        # 先记录Top-3用于诊断
        for idx, search_result in enumerate(search_results[:3]):
            db_answer = str(search_result.get('answer', ''))
            db_question = str(search_result.get('question', ''))
            score = search_result.get('score', 0)
            
            result['top_results'].append({
                'rank': idx + 1,
                'score': round(score, 4),
                'question': db_question[:100] if len(db_question) > 100 else db_question,
                'answer': db_answer[:100] if len(db_answer) > 100 else db_answer,
                'answer_match': self._exact_match(expected_answer, db_answer)
            })
        
        # 再检查各个recall@k的命中情况
        for k in recall_k_values:
            hit_found = False
            for search_result in search_results[:k]:
                db_answer = str(search_result.get('answer', ''))
                if self._exact_match(expected_answer, db_answer):
                    hit_found = True
                    break
            result[f'hit_at_{k}'] = hit_found
        
        return result
    
    def _generate_report(self, test_results: List[Dict[str, Any]],
                        metrics: Dict[str, float], test_config: Dict[str, Any],
                        start_time: datetime, end_time: datetime,
                        initial_metrics: Dict[str, Any], initial_gpu_metrics: Dict[str, Any],
                        final_metrics: Dict[str, Any], final_gpu_metrics: Dict[str, Any]) -> str:
        """
        生成测试报告
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_dir = self.test_config['test']['report_output_dir']
        os.makedirs(report_dir, exist_ok=True)
        
        report_path = os.path.join(report_dir, f"qa_test_report_{timestamp}.json")
        
        duration_seconds = (end_time - start_time).total_seconds()
        
        report_data = {
            "test_info": {
                "timestamp": timestamp,
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "duration_seconds": round(duration_seconds, 2),
                "test_data_path": test_config.get('test_csv_path'),
                "total_queries": len(test_results),
                "top_k": test_config.get('top_k', self.test_config['vector_search']['top_k'])
            },
            "metrics": metrics,
            "performance": {
                "initial": {
                    "cpu_memory": initial_metrics,
                    "gpu": initial_gpu_metrics
                },
                "final": {
                    "cpu_memory": final_metrics,
                    "gpu": final_gpu_metrics
                },
                "summary": {
                    "cpu_usage_increase": round(final_metrics.get('cpu_percent', 0) - initial_metrics.get('cpu_percent', 0), 2),
                    "memory_usage_increase": round(final_metrics.get('memory_percent', 0) - initial_metrics.get('memory_percent', 0), 2)
                }
            },
            "config": test_config,
            "detailed_results": test_results
        }
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"测试报告已保存: {report_path}")
        return report_path
    
    async def run_test(self, request: TestRequest) -> TestResponse:
        """
        运行测试
        """
        try:
            start_time = datetime.now()

            # 收集测试开始时的性能数据
            initial_metrics = self.metrics_collector._get_cpu_memory_metrics()
            initial_gpu_metrics = self.metrics_collector._get_gpu_metrics()

            test_csv_path = request.test_csv_path or self.test_config['test_data']['input_csv_path']
            top_k = request.top_k or self.test_config['vector_search']['top_k']
            recall_k_values = request.recall_k_values or self.test_config['test']['recall_k_values']
            max_workers = self.test_config['test'].get('max_workers', 4)
            
            test_df = self._load_test_data(test_csv_path)
            
            try:
                if self.store.collection is None:
                    self.store.connect()
                    self.store.create_collection(drop_existing=False)
                self.store.load()
            except Exception as e:
                self.logger.warning(f"Milvus连接检查失败，尝试重新连接: {str(e)}")
                self.store.connect()
                self.store.create_collection(drop_existing=False)
                self.store.load()
            
            test_results = []
            progress_interval = self.test_config['test'].get('batch_size', 100)
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                for _, row in test_df.iterrows():
                    future = executor.submit(self._process_single_query, row, top_k, recall_k_values)
                    futures.append(future)
                
                completed = 0
                for future in as_completed(futures):
                    result = future.result()
                    test_results.append(result)
                    completed += 1
                    if completed % progress_interval == 0:
                        self.logger.info(f"已处理 {completed}/{len(test_df)} 个查询")
            
            end_time = datetime.now()

            # 收集测试结束时的性能数据
            final_metrics = self.metrics_collector._get_cpu_memory_metrics()
            final_gpu_metrics = self.metrics_collector._get_gpu_metrics()

            metrics = self._calculate_recall_metrics(test_results, recall_k_values)
            exact_matches = sum(1 for result in test_results if result.get('hit_at_1', False))
            
            test_config_for_report = {
                'test_csv_path': test_csv_path,
                'top_k': top_k,
                'recall_k_values': recall_k_values
            }
            
            report_path = self._generate_report(test_results, metrics, test_config_for_report, start_time, end_time,
                                               initial_metrics, initial_gpu_metrics, final_metrics, final_gpu_metrics)
            
            test_metrics = TestMetrics(
                recall_at_1=metrics.get('recall_at_1', 0.0),
                recall_at_3=metrics.get('recall_at_3', 0.0),
                recall_at_5=metrics.get('recall_at_5', 0.0),
                total_queries=len(test_results),
                exact_matches=exact_matches
            )
            
            duration = (end_time - start_time).total_seconds()
            
            return TestResponse(
                success=True,
                message=f"测试完成，处理了 {len(test_results)} 个查询，耗时 {duration:.2f} 秒",
                metrics=test_metrics,
                report_path=report_path,
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            self.logger.error(f"测试执行失败: {str(e)}")
            raise HTTPException(status_code=500, detail=f"测试执行失败: {str(e)}")
    
    async def run_test_with_uploaded_file(self, file: UploadFile, 
                                        top_k: Optional[int] = None,
                                        recall_k_values: Optional[List[int]] = None) -> TestResponse:
        """
        使用上传文件运行测试
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_file_path = f"temp/uploaded_test_{timestamp}.csv"
            
        os.makedirs("temp", exist_ok=True)
            
        content = await file.read()
        with open(temp_file_path, 'wb') as f:
            f.write(content)
            
        request = TestRequest(
            test_csv_path=temp_file_path,
            top_k=top_k,
            recall_k_values=recall_k_values
        )
            
        response = await self.run_test(request)
            
        os.remove(temp_file_path)
            
        return response
