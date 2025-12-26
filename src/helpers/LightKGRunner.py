# -*- coding: UTF-8 -*-
import torch
import numpy as np
import logging
from helpers.BaseRunner import BaseRunner
from models.BaseModel import BaseModel
from typing import Dict, List

class LightKGRunner(BaseRunner):
    
    def evaluate(self, dataset: BaseModel.Dataset, topks: list, metrics: list) -> Dict[str, float]:
        """
        重写 evaluate 流程：
        1. predict: 获取原始 [Batch, N_Item] 分数，并进行 Masking (排除历史点击，保留 Target)
        2. swap: 将 Target 的分数交换到第 0 列 (满足 evaluate_method 要求)
        3. evaluate_method: 计算指标
        """
        # 1. 获取所有物品的预测分数 [Batch_Size, N_Items]
        # 注意：这里依然会计算全量分数，但只取我们需要的部分进行评估
        predictions = self.predict(dataset)
        
        # 2. 根据模式处理分数
        if dataset.model.test_all:
            # =========== 模式 A: 全量测评 (保持原有逻辑) ===========
            targets = np.array(dataset.data['item_id'])
            if len(targets.shape) > 1: targets = targets[:, 0]
            
            rows = np.arange(len(predictions))
            target_cols = targets.astype(int)
            
            # 交换 Target 到第 0 列
            col0_vals = predictions[:, 0].copy()
            target_vals = predictions[rows, target_cols].copy()
            predictions[:, 0] = target_vals
            predictions[rows, target_cols] = col0_vals
            
        else:
            # =========== 模式 B: 采样测评 (新增逻辑) ===========
            # 此时我们需要构建一个 [Batch, 1 + K] 的小矩阵
            # 第 0 列是 Target，后面是 neg_items
            
            targets = dataset.data['item_id'] # List [Batch]
            neg_items = dataset.data['neg_items'] # List of List [Batch, 100]
            
            # 将 targets 和 neg_items 拼成一个索引矩阵
            test_indices = []
            for i in range(len(targets)):
                # 确保 target 是 int
                t = targets[i] if isinstance(targets[i], (int, np.integer)) else targets[i][0]
                # 拼接 [target, neg1, neg2, ...]
                row_indices = [t] + list(neg_items[i])
                test_indices.append(row_indices)
            
            test_indices = np.array(test_indices) # Shape: [Batch, 101]
            
            # 使用 numpy 高级索引从大矩阵中提取分数
            # row_idx: [[0,0...], [1,1...]]
            row_idx = np.arange(len(predictions))[:, None]
            # 提取后的 predictions 只有 101 列，且第 0 列已经是 Target
            predictions = predictions[row_idx, test_indices]

        # 3. 计算指标 (evaluate_method 默认认为第0列是正确答案，这在上面两种模式下都已满足)
        return self.evaluate_method(predictions, topks, metrics)
 

    def predict(self, dataset: BaseModel.Dataset, save_prediction: bool = False) -> np.ndarray:
        # 复用 BaseRunner 的 predict 流程，但增强 Masking 逻辑
        # 必须确保不 Mask 掉当前的 Target Item
        dataset.model.eval()
        predictions = list()
        
        # 减小 Batch Size 防止 OOM
        eval_bs = 32 if dataset.model.test_all else self.eval_batch_size
        
        from torch.utils.data import DataLoader
        from utils import utils
        from tqdm import tqdm
        
        dl = DataLoader(dataset, batch_size=eval_bs, shuffle=False, num_workers=self.num_workers,
                        collate_fn=dataset.collate_batch, pin_memory=self.pin_memory)
        
        for batch in tqdm(dl, leave=False, ncols=100, mininterval=1, desc='Predict'):
            prediction = dataset.model.inference(utils.batch_to_gpu(batch, dataset.model.device))['prediction']
            predictions.extend(prediction.cpu().data.numpy())
        
        predictions = np.array(predictions)
        
        if dataset.model.test_all:
            rows, cols = list(), list()
            for i, u in enumerate(dataset.data['user_id']):
                # 获取历史点击
                clicked_items = dataset.corpus.train_clicked_set.get(u, set()) | \
                                dataset.corpus.residual_clicked_set.get(u, set())
                
                # 获取当前 Target (绝对不能 Mask)
                target_item = dataset.data['item_id'][i]
                if isinstance(target_item, np.ndarray): target_item = target_item[0]
                
                for item in clicked_items:
                    if item != target_item:
                        rows.append(i)
                        cols.append(item)
            
            if len(rows) > 0:
                predictions[rows, cols] = -np.inf
                
        return predictions

    @staticmethod
    def evaluate_method(predictions: np.ndarray, topk: list, metrics: list) -> Dict[str, float]:
        evaluations = dict()
        # 假设第0列是 GT，计算 Rank
        gt_rank = (predictions > predictions[:,0].reshape(-1,1)).sum(axis=-1) + 1
        
        for k in topk:
            hit = (gt_rank <= k)
            for metric in metrics:
                key = '{}@{}'.format(metric, k)
                if metric in ['HR', 'HIT', 'RECALL', 'PRECISION']:
                    val = hit.mean()
                    if metric == 'PRECISION': val /= k
                    evaluations[key] = val
                elif metric == 'NDCG':
                    evaluations[key] = (hit / np.log2(gt_rank + 1)).mean()
                elif metric == 'MRR':
                    rr = 1.0 / gt_rank
                    rr[gt_rank > k] = 0.0
                    evaluations[key] = rr.mean()
        return evaluations