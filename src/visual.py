# -*- coding: UTF-8 -*-
import torch
import torch.nn as nn
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import importlib
from collections import defaultdict
import math

# ==============================================================================
#  1. 配置区域：六大门派集结 (请确认路径无误)
# ==============================================================================
MODEL_PATHS = {
    'LightKG (Ours)': '../model/LightKG2/LightKG2__Grocery_and_Gourmet_Food__0__lr=0.005__l2=5e-05__emb_size=64__n_layers=2__mess_dropout=0.1__cos_loss=1__user_loss=1e-08__item_loss=1e-07.pt',
    'LightGCN':'../model/LightGCN/LightGCN__Grocery_and_Gourmet_Food__0__lr=0.001__l2=1e-08__emb_size=64__n_layers=3__batch_size=2048.pt',
    'NeuMF':'../model/NeuMF/NeuMF__Grocery_and_Gourmet_Food__0__lr=0.0005__l2=1e-07__emb_size=64__layers=[64].pt',
    'DirectAU':'../model/DirectAU/DirectAU__Grocery_and_Gourmet_Food__0__lr=0.001__l2=1e-05__emb_size=64__gamma=0.3.pt',
    'BUIR':'../model/BUIR/BUIR__Grocery_and_Gourmet_Food__0__lr=0.001__l2=1e-06__emb_size=64__momentum=0.995.pt',
    'BPRMF':'../model/BPRMF/BPRMF__Grocery_and_Gourmet_Food__0__lr=0.001__l2=1e-06__emb_size=64__batch_size=2048.pt'
}

DATA_ROOT = '../data/Grocery_and_Gourmet_Food/Grocery_and_Gourmet_Food'
META_FILE = '../data/Grocery_and_Gourmet_Food/Grocery_and_Gourmet_Food/meta_Grocery_and_Gourmet_Food.json'
TRAIN_FILE = '../data/Grocery_and_Gourmet_Food/train.csv'

TARGET_CATS = ['Coffee', 'Candy', 'Baby Foods', 'Cereal', 'Cooking & Baking']

# ==============================================================================
#  2. 全能参数包 (根据你上传的文件精确匹配)
# ==============================================================================
class Args:
    # 基础参数
    emb_size = 64
    dataset = 'Grocery_and_Gourmet_Food'
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    # 通用参数
    lr = 1e-3
    l2 = 1e-5
    batch_size = 2048
    dropout = 0
    num_neg = 1
    test_all = 0
    
    # LightGCN / LightKG
    n_layers = 3        
    mess_dropout = 0.1 
    cos_loss = 1       
    user_loss = 1e-8   
    item_loss = 1e-7
    k_neighbors = 10
    
    # NeuMF 特有 (必须是字符串，因为源码用了 eval)
    layers = '[64]'       
    
    # DirectAU 特有
    gamma = 0.3         
    
    # BUIR 特有
    momentum = 0.995    
    
    # 其他防报错参数
    temperature = 0.2
    model_path = ''     
    log_file = '/tmp/log.txt' 
    train = 0           
    topk = '5,10,20'
    metric = 'NDCG,HR'
    main_metric = ''
    optimizer = 'Adam'
    epoch = 100
    check_epoch = 1
    test_epoch = 1
    early_stop = 10
    reg = 1e-5
    max_len = 20
    num_workers = 0
    eval_batch_size = 256
    pin_memory = 0
    buffer = 1

# ==============================================================================
#  3. 核心工具函数
# ==============================================================================
def get_stats():
    print(">>> 1. 读取训练集计算 User/Item 数量...")
    try:
        df = pd.read_csv(TRAIN_FILE, sep='\t')
    except:
        df = pd.read_csv(os.path.join(os.path.dirname(DATA_ROOT), '../train.csv'), sep='\t')
    n_users = df['user_id'].max() + 1
    n_items = df['item_id'].max() + 1
    return n_users, n_items

def load_meta_categories():
    print(">>> 2. 加载元数据和类别信息...")
    with open(os.path.join(DATA_ROOT, 'item2newid.json'), 'r') as f:
        asin2int = json.load(f)
    valid_asins = set(asin2int.keys())
    item_cats = {}
    target_map = {t.lower(): t for t in TARGET_CATS}
    
    with open(META_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                item = eval(line)
                asin = item.get('asin')
                if asin in valid_asins:
                    iid = asin2int[asin]
                    cats_list = item.get('categories', [['Unknown']])
                    found_cat = None
                    for c_chain in cats_list:
                        for c in c_chain:
                            if c.lower() in target_map:
                                found_cat = target_map[c.lower()]
                                break
                        if found_cat: break
                    if found_cat:
                        item_cats[iid] = found_cat
            except: pass
    print(f"   已匹配到目标类别的商品数: {len(item_cats)}")
    return item_cats

def get_corpus(n_users, n_items):
    class Corpus:
        def __init__(self):
            self.n_users = n_users
            self.n_items = n_items
            self.n_relations = 9 
            self.n_entities = n_items + 100
            self.item_meta_df = None
            self.train_clicked_set = defaultdict(list)
            self.residual_clicked_set = defaultdict(list)
            self.relation_df = pd.DataFrame({'head': [0], 'tail': [0], 'relation': [0]})
    return Corpus()

def extract_embedding_smartly(model, corpus, model_name):
    # 1. NeuMF (拼接 GMF 和 MLP)
    if hasattr(model, 'mf_i_embeddings') and hasattr(model, 'mlp_i_embeddings'):
        # print(f"   ✅ {model_name}: 识别为 NeuMF 结构")
        gmf = model.mf_i_embeddings.weight
        mlp = model.mlp_i_embeddings.weight
        return torch.cat([gmf, mlp], dim=1)
    
    # 2. LightGCN (Encoder 里的 embedding_dict)
    if hasattr(model, 'encoder') and hasattr(model.encoder, 'embedding_dict'):
        emb_dict = model.encoder.embedding_dict
        if 'item_emb' in emb_dict: return emb_dict['item_emb']

    # 3. DirectAU / BPRMF (都叫 i_embeddings)
    if hasattr(model, 'i_embeddings'):
        # print(f"   ✅ {model_name}: 找到 i_embeddings")
        return model.i_embeddings.weight

    # 4. BUIR (叫 item_online)
    if hasattr(model, 'item_online'):
        # print(f"   ✅ {model_name}: 找到 item_online")
        if isinstance(model.item_online, nn.Embedding):
            return model.item_online.weight
        elif hasattr(model.item_online, 'weight'): # 处理被封装的情况
            return model.item_online.weight

    # 5. LightKG (entity_embedding)
    if hasattr(model, 'entity_embedding'):
        return model.entity_embedding.weight

    # 6. 兜底策略
    if hasattr(model, 'embedding'): 
        return model.embedding.weight[corpus.n_users : corpus.n_users + corpus.n_items]
        
    return None

def load_model_and_extract(model_name, path, corpus):
    print(f">>> 加载模型: {model_name}...")
    if not os.path.exists(path):
        print(f"⚠️ 文件不存在: {path}")
        return None

    real_model_name = model_name.split(' ')[0] 
    
    try:
        chkpt = torch.load(path, map_location=Args.device)
        state_dict = chkpt['model_state_dict'] if 'model_state_dict' in chkpt else chkpt
        
        if 'relation_embedding.weight' in state_dict:
            real_rel_num = state_dict['relation_embedding.weight'].shape[0]
            corpus.n_relations = (real_rel_num + 1) // 2
        if 'entity_embedding.weight' in state_dict:
            corpus.n_entities = state_dict['entity_embedding.weight'].shape[0]

        # 动态导入
        try:
            module = importlib.import_module(f"models.general.{real_model_name}")
            ModelClass = getattr(module, real_model_name)
        except ImportError:
             if 'LightKG' in real_model_name:
                module = importlib.import_module("models.general.LightKG")
                ModelClass = getattr(module, 'LightKG')
             else:
                raise

        model = ModelClass(Args, corpus).to(Args.device)
        model.load_state_dict(state_dict, strict=False) 
        
        emb = extract_embedding_smartly(model, corpus, model_name)
        if emb is None:
            print(f"❌ {model_name}: 无法提取 Embedding")
            # 打印属性名以供调试
            # print(f"   属性: {list(model.__dict__['_modules'].keys())}")
            return None
            
        return emb.detach().cpu().numpy()

    except Exception as e:
        print(f"❌ {model_name} 加载失败: {e}")
        return None

def run_multi_compare():
    n_users, n_items = get_stats()
    item_cats = load_meta_categories()
    corpus = get_corpus(n_users, n_items)
    
    # 筛选与采样
    df = pd.DataFrame(list(item_cats.items()), columns=['iid', 'category'])
    df_sub = df.groupby('category').apply(lambda x: x.sample(min(len(x), 300), random_state=42)).reset_index(drop=True)
    ids = df_sub['iid'].values
    cats = df_sub['category'].values
    
    print("\n--- 开始批量降维 (t-SNE) ---")
    tsne = TSNE(n_components=2, init='pca', learning_rate='auto', perplexity=40, random_state=2023)
    
    plot_data = []
    
    for display_name, path in MODEL_PATHS.items():
        vecs_all = load_model_and_extract(display_name, path, corpus)
        
        if vecs_all is not None:
            current_n_items = vecs_all.shape[0]
            valid_mask = ids < current_n_items
            valid_ids = ids[valid_mask]
            valid_cats = cats[valid_mask]
            
            if len(valid_ids) < 100:
                print(f"⚠️ {display_name} 有效 Item ID 太少，跳过")
                continue
                
            print(f"   🌀 计算 t-SNE: {display_name} ...")
            vecs_sub = vecs_all[valid_ids]
            vecs_2d = tsne.fit_transform(vecs_sub)
            
            plot_data.append({'name': display_name, 'vecs': vecs_2d, 'cats': valid_cats})

    if not plot_data: return

    # 绘图
    num_models = len(plot_data)
    cols = 3
    rows = math.ceil(num_models / cols)
    
    print(f"\n>>> 绘图 (布局: {rows}x{cols})...")
    fig, axes = plt.subplots(rows, cols, figsize=(cols*6, rows*5))
    if num_models > 1: axes = axes.flatten()
    else: axes = [axes]
    
    palette = sns.color_palette("bright", len(set(cats)))
    
    for i, data in enumerate(plot_data):
        ax = axes[i]
        sns.scatterplot(x=data['vecs'][:,0], y=data['vecs'][:,1], hue=data['cats'], 
                        palette=palette, s=40, alpha=0.6, ax=ax, legend=False)
        ax.set_title(data['name'], fontsize=16, fontweight='bold')
        ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel(''); ax.set_ylabel('')

    for j in range(i + 1, len(axes)): axes[j].axis('off')
        
    lines = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=c, markersize=10) for c in palette]
    labels = sorted(list(set(cats)))
    fig.legend(lines, labels, loc='lower center', ncol=len(labels), bbox_to_anchor=(0.5, 0.02), fontsize=14, title="Category")
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)
    
    save_path = 'vis_multi_model_final.png'
    plt.savefig(save_path, dpi=300)
    print(f"\n✅ 成功！已保存: {save_path}")

if __name__ == '__main__':
    run_multi_compare()