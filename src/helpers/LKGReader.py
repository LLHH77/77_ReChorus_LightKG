# -*- coding: UTF-8 -*-
import os
import logging
import pandas as pd
import numpy as np
import ast
from helpers.BaseReader import BaseReader

class LKGReader(BaseReader):
    @staticmethod
    def parse_data_args(parser):
        parser.add_argument('--use_meta_as_kg', type=int, default=1, 
                            help='Whether to build KG from item_meta.csv columns.')
        return BaseReader.parse_data_args(parser)

    def __init__(self, args):
        # 1. 调用父类初始化，这会执行 _read_data() 并计算 self.n_items
        super().__init__(args)
        
        # 2. 初始化 KG 相关属性
        self.n_entities = self.n_items
        self.n_relations = 2 
        self.relation_df = pd.DataFrame(columns=['head', 'tail', 'relation'])

        # 3. 构建 KG
        self._construct_kg(args)

    def _construct_kg(self, args):
        dataset_dir = os.path.join(self.prefix, self.dataset)
        kg_file = os.path.join(dataset_dir, f'{self.dataset}.kg')
        
        # 策略 1: 优先寻找标准的 .kg 文件
        if os.path.exists(kg_file):
            logging.info(f"Found explicit KG file: {kg_file}")
            self._load_explicit_kg(kg_file)
        # 策略 2: 尝试从 item_meta.csv 构建
        elif args.use_meta_as_kg:
            meta_file = os.path.join(dataset_dir, 'item_meta.csv')
            if os.path.exists(meta_file):
                logging.info(f"Building KG from metadata: {meta_file}")
                self._build_kg_from_meta(meta_file)
            else:
                logging.warning("No KG file or Item Meta file found. Running without KG.")
        else:
            logging.info("Skipping KG construction.")

    def _load_explicit_kg(self, kg_path):
        # 读取标准 KG 文件 (格式: head_id \t relation_id \t tail_id)
        kg_df = pd.read_csv(kg_path, sep='\t', names=['head', 'relation', 'tail'])
        self.relation_df = kg_df
        # 确保 n_relations 足够大
        current_max_rel = kg_df['relation'].max() if not kg_df.empty else 0
        self.n_relations = max(self.n_relations, current_max_rel + 2)
        
        current_max_ent = max(kg_df['head'].max(), kg_df['tail'].max()) if not kg_df.empty else 0
        self.n_entities = max(self.n_entities, current_max_ent + 1)

    def _build_kg_from_meta(self, meta_path):
        try:
            meta_df = pd.read_csv(meta_path, sep='\t')
        except:
            meta_df = pd.read_csv(meta_path, sep=',') # 兼容逗号分隔

        if 'item_id' not in meta_df.columns:
            raise ValueError("item_meta.csv must have 'item_id' column")
        
        triples = []
        current_entity_id = self.n_items
        relation_cnt = 0
        attr_value_map = {} 

        # 遍历每一列
        for col in meta_df.columns:
            if col == 'item_id':
                continue
            
            # --- Case 1: 属性列 (i_) ---
            if col.startswith('i_'):
                relation_id = relation_cnt
                relation_cnt += 1
                logging.info(f"Mapping Attribute Column '{col}' to Relation ID {relation_id}")
                
                # 获取该列所有非空且有效的值
                temp_df = meta_df[['item_id', col]].dropna()
                
                for _, row in temp_df.iterrows():
                    try:
                        iid = int(row['item_id'])
                        # [关键] 过滤掉不在交互矩阵里的 item，防止越界
                        # 注意：如果这里的 item_id 是原始 ID，而 self.n_items 是映射后的数量，
                        # 且没有做 ID 映射，这里会导致所有数据被过滤。
                        if iid >= self.n_items: 
                            continue

                        val = row[col]
                        key = f"{col}_{val}"
                        if key not in attr_value_map:
                            attr_value_map[key] = current_entity_id
                            current_entity_id += 1
                        
                        eid = attr_value_map[key]
                        triples.append([iid, eid, relation_id])
                    except ValueError:
                        continue

            # --- Case 2: 物品关系列 (r_) ---
            elif col.startswith('r_'):
                relation_id = relation_cnt
                relation_cnt += 1
                logging.info(f"Mapping Item-Item Relation '{col}' to Relation ID {relation_id}")
                
                for _, row in meta_df.iterrows():
                    try:
                        iid = int(row['item_id'])
                        if iid >= self.n_items:
                            continue

                        val_str = row[col]
                        if pd.isna(val_str) or not isinstance(val_str, str) or len(val_str) < 2:
                            continue
                        
                        if val_str.strip().startswith('['):
                            # 使用 literal_eval 代替 eval，更安全
                            target_items = ast.literal_eval(val_str)
                            for target_id in target_items:
                                target_id = int(target_id)
                                # 确保尾实体也在有效范围内
                                if 0 <= target_id < self.n_items:
                                    triples.append([iid, target_id, relation_id])
                    except (ValueError, SyntaxError):
                        continue

        # [修复] 汇总信息时的空值处理
        self.n_entities = current_entity_id
        # 确保 n_relations 至少为 2，且比 relation_cnt 大（留出 buffer）
        self.n_relations = max(2, relation_cnt + 2)
        
        if len(triples) > 0:
            self.relation_df = pd.DataFrame(triples, columns=['head', 'tail', 'relation'])
        else:
            logging.warning("!!! WARNING: No triples constructed from item_meta.csv. "
                            "Please check if 'item_id' in meta file matches the internal IDs in dataset.")
            self.relation_df = pd.DataFrame(columns=['head', 'tail', 'relation'])
        
        logging.info(f"KG Constructed. #Items: {self.n_items}, #Entities: {self.n_entities}, #Relations: {self.n_relations}")
        logging.info(f"Triples count: {len(self.relation_df)}")