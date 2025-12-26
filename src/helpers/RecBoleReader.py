# -*- coding: UTF-8 -*-
import os
import logging
import numpy as np
import pandas as pd
from helpers.BaseReader import BaseReader

class RecBoleReader(BaseReader):
    @staticmethod
    def parse_data_args(parser):
        parser.add_argument('--recbole_format', type=int, default=1, help='Whether dataset is in RecBole format.')
        return BaseReader.parse_data_args(parser)

    def __init__(self, args):
        self.sep = args.sep
        self.prefix = args.path
        self.dataset = args.dataset
        
        # 1. 读取交互数据
        self._read_recbole_inter()
        
        # 2. 构建点击集合
        self.train_clicked_set = dict()  # 用户在训练集中点过的 item
        self.residual_clicked_set = dict() # 用户在 dev/test 中出现过的 item
        for key in ['train', 'dev', 'test']:
            df = self.data_df[key]
            for uid, iid in zip(df['user_id'], df['item_id']):
                if uid not in self.train_clicked_set:
                    self.train_clicked_set[uid] = set()
                    self.residual_clicked_set[uid] = set()
                if key == 'train':
                    self.train_clicked_set[uid].add(iid)
                else:
                    self.residual_clicked_set[uid].add(iid)
        
        # 3. 构建 KG
        self._construct_kg()

    def _read_recbole_inter(self):
        logging.info(f'Reading RecBole data from {self.prefix}/{self.dataset}')
        self.data_df = dict()
        phase_map = {'train': 'train', 'dev': 'valid', 'test': 'test'}
        
        all_dfs = []
        for phase, recbole_phase in phase_map.items():
            file_path = os.path.join(self.prefix, self.dataset, f'{self.dataset}.{recbole_phase}.inter')
            if not os.path.exists(file_path):
                file_path = os.path.join(self.prefix, self.dataset, f'{self.dataset}.{phase}.inter')
            
            if os.path.exists(file_path):
                df = pd.read_csv(file_path, sep='\t')
                u_col = next(c for c in df.columns if 'user_id' in c)
                i_col = next(c for c in df.columns if 'item_id' in c)
                df = df.rename(columns={u_col: 'user_id', i_col: 'item_id'})
                
                # 简化处理 time/rating
                df['time'] = 0
                df = df[['user_id', 'item_id', 'time']]
                df['phase'] = phase
                all_dfs.append(df)
            else:
                raise FileNotFoundError(f"Inter file not found for {phase}")
        
        full_df = pd.concat(all_dfs)
        unique_users = sorted(full_df['user_id'].unique())
        unique_items = sorted(full_df['item_id'].unique())
        
        self.user_id_map = {uid: i for i, uid in enumerate(unique_users)}
        self.item_id_map = {iid: i for i, iid in enumerate(unique_items)}
        self.n_users = len(unique_users)
        self.n_items = len(unique_items)
        
        logging.info(f"# Users: {self.n_users}, # Items: {self.n_items}")
        
        full_df['user_id'] = full_df['user_id'].map(self.user_id_map)
        full_df['item_id'] = full_df['item_id'].map(self.item_id_map)
        
        for phase in phase_map:
            self.data_df[phase] = full_df[full_df['phase'] == phase].drop(columns=['phase']).reset_index(drop=True)
        self.all_df = pd.concat([self.data_df[key] for key in ['train', 'dev', 'test']])

    def _construct_kg(self):
        logging.info("Constructing Knowledge Graph...")
        link_file = os.path.join(self.prefix, self.dataset, f'{self.dataset}.link')
        kg_file = os.path.join(self.prefix, self.dataset, f'{self.dataset}.kg')
        
        if not os.path.exists(kg_file):
            # 情况 ———— 没有 KG 文件
            self.n_entities = self.n_items
            self.n_relations = 2 # 至少保留 User-Item 的空间
            self.relation_df = pd.DataFrame()
            return

        # 1. Entity ID 映射
        self.entity_id_map = {}
        if os.path.exists(link_file):
            # 如果一个 item 已经对应某个 entity，那这个 entity 的 ID 直接等于 item ID
            # 保证 item embedding 与 entity embedding 对齐
            link_df = pd.read_csv(link_file, sep='\t')
            i_col = next(c for c in link_df.columns if 'item_id' in c)
            e_col = next(c for c in link_df.columns if 'entity_id' in c)
            for _, row in link_df.iterrows():
                if row[i_col] in self.item_id_map:
                    self.entity_id_map[row[e_col]] = self.item_id_map[row[i_col]]
        
        kg_df = pd.read_csv(kg_file, sep='\t')
        h_col = next(c for c in kg_df.columns if 'head_id' in c)
        t_col = next(c for c in kg_df.columns if 'tail_id' in c)
        r_col = next(c for c in kg_df.columns if 'relation_id' in c)
        
        all_kg_entities = set(kg_df[h_col]).union(set(kg_df[t_col]))
        current_idx = self.n_items
        for ent in all_kg_entities:
            if ent not in self.entity_id_map:
                self.entity_id_map[ent] = current_idx
                current_idx += 1
        self.n_entities = current_idx
        
        # 2. Relation 处理 (关键修正)
        unique_relations = kg_df[r_col].unique()
        self.relation_id_map = {r: i+1 for i, r in enumerate(unique_relations)}
        # 【关键】n_relations 必须足够大，以容纳 LightKG 的偏移量计算
        # LightKG formula: Offset = n_relations - 2
        # 我们希望 Offset = len(unique_relations)
        # 所以 n_relations = len(unique_relations) + 2
        self.n_relations = len(unique_relations) + 2
        
        kg_df['head'] = kg_df[h_col].map(self.entity_id_map)
        kg_df['tail'] = kg_df[t_col].map(self.entity_id_map)
        kg_df['relation'] = kg_df[r_col].map(self.relation_id_map)
        self.relation_df = kg_df[['head', 'tail', 'relation']].dropna().astype(int)
        
        print("Raw KG relation max:", kg_df[r_col].max())
        print("Mapped relation max:", kg_df['relation'].max())
        
        logging.info(f"KG Stats: # Entities: {self.n_entities}, # Relations: {self.n_relations} (Unique: {len(unique_relations)})")