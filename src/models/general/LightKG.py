# -*- coding: UTF-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.sparse import coo_matrix
import torch_sparse
from models.BaseModel import GeneralModel
import logging

class LightKG(GeneralModel):
    reader = 'RecBoleReader'
    runner = 'LightKGRunner'
    extra_log_args = ['emb_size', 'n_layers', 'mess_dropout', 'cos_loss', 'user_loss', 'item_loss']

    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--emb_size', type=int, default=64, help='Size of embedding vectors.')
        parser.add_argument('--n_layers', type=int, default=2, help='Number of GCN layers.')
        parser.add_argument('--mess_dropout', type=float, default=0.1, help='Dropout rate.')
        parser.add_argument('--cos_loss', type=int, default=1, help='Use contrastive loss.')
        parser.add_argument('--user_loss', type=float, default=1e-08, help='Weight for user CL.')
        parser.add_argument('--item_loss', type=float, default=1e-07, help='Weight for item CL.')
        parser.add_argument('--temperature', type=float, default=0.6, help='Temperature for CL.')
        return GeneralModel.parse_model_args(parser)

    def __init__(self, args, corpus):
        super().__init__(args, corpus)
        if hasattr(args, 'test_all'):
            self.test_all = args.test_all
        else:
            self.test_all = 0
        
        self.emb_size = args.emb_size
        self.n_layers = args.n_layers
        self.mess_dropout_rate = args.mess_dropout
        self.cos_loss = args.cos_loss
        self.beta_u = args.user_loss
        self.beta_i = args.item_loss
        self.temperature = args.temperature
        
        self.n_entities = corpus.n_entities
        self.n_relations = corpus.n_relations

        self._build_graphs(corpus)
        self._define_params()
        self.apply(self.init_weights)
        self.test = False
        
    def init_weights(self, m):
        """
        覆盖父类的初始化方法，使用 Xavier Uniform 以对齐原代码
        Reference: https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.xavier_normal_
        """
        # 对 Embedding 层进行 Xavier Normal 初始化
        if isinstance(m, nn.Embedding):
            nn.init.xavier_uniform_(m.weight)  
        
        # 对 Linear 层进行 Xavier Normal 初始化
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight) 
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def _build_graphs(self, corpus):
        self._build_interaction_matrix(corpus)
        self._build_kg_graph(corpus)
        self._build_ckg()  # 不返回 sparse tensor，而是设置 self.CKG_indices 和 self.CKG_values
        print("CKG after build...")
        print("CKG max relation id after build:", self.CKG_values.max().item()) 
        self.Degree = self._build_degree_matrix()
        self.Similarity_matrix = self._build_similarity_matrix()
        print("CKG object id:", id(self))
        print("CKG storage ptr:", self.CKG_values.data_ptr()) 

    def _build_interaction_matrix(self, corpus):
        rows, cols = [], []
        for user_id in corpus.train_clicked_set:
            for item_id in corpus.train_clicked_set[user_id]:
                rows.append(user_id)
                cols.append(item_id)
        self.inter = coo_matrix((np.ones(len(rows)), (rows, cols)), 
                                shape=(corpus.n_users, corpus.n_items), dtype=np.float32)

    def _build_kg_graph(self, corpus):
        if hasattr(corpus, 'relation_df') and len(corpus.relation_df) > 0:
            heads = corpus.relation_df['head'].values
            tails = corpus.relation_df['tail'].values
            relations = corpus.relation_df['relation'].values
            self.kg_graph = coo_matrix((relations, (heads, tails)), 
                                       shape=(self.n_entities, self.n_entities), dtype=np.int32)
        else:
            self.kg_graph = coo_matrix((self.n_entities, self.n_entities), dtype=np.int32)
        
        assert corpus.relation_df['relation'].max() < self.n_relations, f"relation overflow before graph: max={corpus.relation_df['relation'].max()}, n_rel={self.n_relations}"
        
    def _build_ckg(self):
        inter_head, inter_tail = self.inter.row, self.inter.col
        kg_head, kg_tail, kg_data = self.kg_graph.row, self.kg_graph.col, self.kg_graph.data
        
        assert kg_data.max() < self.n_relations, f"KG relation overflow: max={kg_data.max()}, n_rel={self.n_relations}"
        
        r_offset = self.n_relations - 2
        
        kg_rel_rev = kg_data + r_offset
        kg_bi_h = np.concatenate([kg_head, kg_tail])
        kg_bi_t = np.concatenate([kg_tail, kg_head])
        kg_bi_r = np.concatenate([kg_data, kg_rel_rev])
        
        u_i_rel = np.array([r_offset * 2 + 1] * len(inter_head))
        i_u_rel = np.array([r_offset * 2 + 2] * len(inter_head))
        
        all_head = np.concatenate([inter_head, inter_tail + self.user_num, kg_bi_h + self.user_num])
        all_tail = np.concatenate([inter_tail + self.user_num, inter_head, kg_bi_t + self.user_num])
        all_rel = np.concatenate([u_i_rel, i_u_rel, kg_bi_r])
        
        # 如果不需要多重边，去重
        # unique_edges = set(zip(all_head, all_tail, all_rel))
        # if unique_edges:
        #     all_head, all_tail, all_rel = zip(*unique_edges)
        # all_head, all_tail, all_rel = np.array(all_head), np.array(all_tail), np.array(all_rel)
        
        self.CKG_indices = torch.tensor(np.stack([all_head, all_tail]), dtype=torch.long).to(self.device)  # 存储 raw indices
        self.CKG_values = torch.tensor(all_rel, dtype=torch.long).to(self.device)  # 存储 raw values
        self.ckg_size = torch.Size([self.n_entities + self.user_num, self.n_entities + self.user_num])
        
        print("Building CKG...")
        print("CKG nnz:", len(self.CKG_values)) 
        print("CKG indices max:", self.CKG_indices.max().item())
        print("CKG max relation id:", self.CKG_values.max().item())
        print("CKG storage ptr:", self.CKG_values.data_ptr())

    def _build_degree_matrix(self):
        inter_idx = torch.tensor(np.stack([self.inter.row, self.inter.col]), dtype=torch.long)
        inter_sp = torch.sparse_coo_tensor(inter_idx, torch.ones(len(self.inter.row)), size=(self.user_num, self.item_num)).coalesce()
        u_deg = torch.sparse.sum(inter_sp, dim=1).to_dense()
        i_deg_inter = torch.sparse.sum(inter_sp, dim=0).to_dense()
        
        kg_bi_h = np.concatenate([self.kg_graph.row, self.kg_graph.col])
        kg_bi_t = np.concatenate([self.kg_graph.col, self.kg_graph.row])
        edges = list(set(zip(kg_bi_h, kg_bi_t)))
        
        if len(edges) > 0:
            h = torch.tensor([e[0] for e in edges], dtype=torch.long)
            t = torch.tensor([e[1] for e in edges], dtype=torch.long)
            kg_sp = torch.sparse_coo_tensor(torch.stack([h, t]), torch.ones(len(h)), size=(self.n_entities, self.n_entities)).coalesce()
            ent_deg = torch.sparse.sum(kg_sp, dim=1).to_dense()
        else:
            ent_deg = torch.zeros(self.n_entities)
            
        i_deg_total = i_deg_inter + ent_deg[:self.item_num]
        deg = torch.cat([u_deg, i_deg_total, ent_deg[self.item_num:]])
        deg = 1 / torch.sqrt(deg + 1e-10)
        return deg.to(self.device)

    def _build_similarity_matrix(self):
        ckg_idx = self.CKG_indices
        mask = (ckg_idx[0] < self.user_num + self.item_num) & (ckg_idx[1] < self.user_num + self.item_num)
        filt_idx = ckg_idx[:, mask]
        
        inter_mat = torch.sparse_coo_tensor(filt_idx, torch.ones(filt_idx.shape[1], device=self.device), dtype=torch.float32)
        sim_mat = (torch.sparse.mm(inter_mat, inter_mat) + inter_mat).coalesce()
        
        idx, val = sim_mat.indices(), sim_mat.values()
            
        deg_prod = self.Degree[idx[0]] * self.Degree[idx[1]]
        norm_val = -val * deg_prod
        
        size = self.user_num + self.item_num
        return torch.sparse_coo_tensor(idx, norm_val, size=(size, size))

    def _define_params(self):
        self.user_embedding = nn.Embedding(self.user_num, self.emb_size)
        self.entity_embedding = nn.Embedding(self.n_entities, self.emb_size)
        self.relation_embedding = nn.Embedding(2 * self.n_relations - 1, 1)
        self.mess_dropout = nn.Dropout(p=self.mess_dropout_rate)

    def _forward(self): 
        # print("Relation Embedding Size:", self.relation_embedding.num_embeddings)
        # print("n_relations:", self.n_relations)
        assert self.CKG_values.max().item() < self.relation_embedding.num_embeddings  # 用 self.CKG_values
        
        u_e = self.user_embedding.weight
        e_e = self.entity_embedding.weight
        if self.mess_dropout_rate > 0 and not self.test:
            u_e = self.mess_dropout(u_e)
            e_e = self.mess_dropout(e_e)
        
        init_emb = torch.cat([u_e, e_e], dim=0)
        layer_embs = [init_emb]
        
        # Relation Embedding Lookup Check
        ckg_vals = self.CKG_values 
        if ckg_vals.max() >= self.relation_embedding.num_embeddings:
             print(f"!!! ERROR: CKG Value {ckg_vals.max()} >= RelEmb Size {self.relation_embedding.num_embeddings}")

        rel_emb = self.relation_embedding(ckg_vals).view(-1)
        edge_idx = self.CKG_indices  
        
        if self.Degree.device != edge_idx.device:
            self.Degree = self.Degree.to(edge_idx.device)

        norm_val = rel_emb * (self.Degree[edge_idx[0]] * self.Degree[edge_idx[1]])
        
        total_size = self.user_num + self.n_entities
        
        # SPMM Check
        if edge_idx.max() >= total_size:
             print(f"!!! ERROR: SPMM Edge Index {edge_idx.max()} >= Total Size {total_size}")
        if init_emb.shape[0] != total_size:
             print(f"!!! ERROR: Init Emb Size {init_emb.shape[0]} != Total Size {total_size}")

        for _ in range(self.n_layers):
            init_emb = torch_sparse.spmm(edge_idx, norm_val, total_size, total_size, init_emb)
            layer_embs.append(init_emb)
            
        agg_emb = torch.mean(torch.stack(layer_embs, dim=1), dim=1)
        return torch.split(agg_emb, [self.user_num, self.n_entities])

    def forward(self, feed_dict):
        self.test = False
        user_ids, item_ids = feed_dict['user_id'], feed_dict['item_id']
        
        self.current_user_ids = user_ids
        if item_ids.dim() > 1:
            self.current_pos_item_ids = item_ids[:, 0]
            self.current_neg_item_ids = item_ids[:, 1:]
        else:
            self.current_pos_item_ids = item_ids
            self.current_neg_item_ids = None

        u_gcn, e_gcn = self._forward()
        self.all_user_embeddings, self.all_entity_embeddings = u_gcn, e_gcn
        
        u_e, i_e = u_gcn[user_ids], e_gcn[item_ids]
        if item_ids.dim() == 1:
             pred = (u_e * i_e).sum(dim=-1)
        else:
             pred = (u_e[:, None, :] * i_e).sum(dim=-1)
        return {'prediction': pred.view(feed_dict['batch_size'], -1)}

    def inference(self, feed_dict):
        user_ids = feed_dict['user_id']
        if not hasattr(self, '_cached_u_emb') or not self.test:
            self.test = True
            with torch.no_grad():
                u_all, e_all = self._forward()
                self._cached_u_emb, self._cached_i_emb = u_all, e_all[:self.item_num]
        
        u_emb = self._cached_u_emb[user_ids]
        i_emb = self._cached_i_emb
        scores = torch.matmul(u_emb, i_emb.t())
        return {'prediction': scores}
    
    def get_user_similarity(self, node):
        """
        Exactly aligned with original LightKG implementation.
        node: Tensor of user indices, shape [B]
        return: Dense similarity matrix [B, B]
        """
        sim = self.Similarity_matrix.index_select(0, node)
        sim = sim.transpose(0, 1)
        sim = sim.index_select(0, node).to_dense()
        sim = 1 + sim
        return sim


    def get_item_similarity(self, p_node, n_node):
        """
        Item-item similarity for contrastive learning (pos-neg).
        p_node, n_node: Tensor of item indices, shape [B]
        """
        p_node = p_node + self.user_num
        n_node = n_node + self.user_num

        sim = self.Similarity_matrix.index_select(0, p_node)
        sim = sim.transpose(0, 1)
        sim = sim.index_select(0, n_node).to_dense()
        sim = 1 + sim
        return sim


    def loss(self, out_dict):
        preds = out_dict['prediction']
        pos, neg = preds[:, 0], preds[:, 1:]
        if neg.shape[1] > 1: pos = pos.unsqueeze(1)
        
        bpr_loss = -torch.log(torch.sigmoid(pos - neg) + 1e-10).mean()
        
        cl_loss = torch.tensor(0.0, device=self.device)
        if self.cos_loss > 0:
            # ---------- User CL ----------
            u_ids = self.current_user_ids
            u_e = self.user_embedding(u_ids)
            
            sim_u = self.get_user_similarity(u_ids)   # [B, B]
            
            norm_u = F.normalize(u_e, p=2, dim=1)
            deg_u = self.Degree[u_ids]
            mat_u = 1 - (deg_u.view(-1, 1) @ deg_u.view(1, -1))
            loss_u = torch.sum(mat_u * torch.exp((norm_u @ norm_u.T) * sim_u / self.temperature))
            
            # ---------- Item CL (pair-wise, aligned with original LightKG) ----------
            p_ids = self.current_pos_item_ids          # [B]
            n_ids = self.current_neg_item_ids[:, 0]    # [B]
            sim = self.get_item_similarity(p_ids, n_ids)
            p_e = self.entity_embedding(p_ids)
            n_e = self.entity_embedding(n_ids)
            norm_p = F.normalize(p_e, p=2, dim=1)
            norm_n = F.normalize(n_e, p=2, dim=1)
            deg_p = self.Degree[p_ids + self.user_num]
            deg_n = self.Degree[n_ids + self.user_num]
            matrix = deg_p.view(-1,1) @ deg_n.view(1,-1)
            matrix = 1 - matrix
            loss_i = torch.sum(matrix * torch.exp((norm_p @ norm_n.T)*sim/self.temperature))
            cl_loss = self.beta_u * loss_u + self.beta_i * loss_i

        return bpr_loss + cl_loss
