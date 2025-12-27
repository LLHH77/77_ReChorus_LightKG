# LightKG based on ReChorus项目解析
 by **赵景琦&&廖桦淇**

项目基于[ReChorus](https://github.com/THUwangcy/ReChorus)框架，复现了LightKG模型，并完成了新框架上的消融实验、对比实验、超参实验和案例分析。LightKG 是一个简约而强大的、基于图神经网络（GNN）的知识图谱感知推荐系统，旨在提高推荐的准确性和训练效率，特别是在交互稀疏的场景下。

> 原论文：[LightKG: Efficient Knowledge-Aware Recommendations with Simplified GNN Architecture ](https://dl.acm.org/doi/abs/10.1145/3711896.3737026)
##  LightKG复现环境配置

已在 Python 3.9 和 Ubuntu 20.04 上经过测试

1. 安装指定版本的包！注意顺序很重要，必须先装torch
```bash
pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118
```

2. 再装torch-scatter
```bash
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
```

3. 安装其它包，注意限制版本
```bash
pip install "numpy<1.24" "recbole==1.1.1" lightgbm xgboost ray thop matplotlib seaborn scipy ipywidgets
```
##  数据集准备

MovieLens_1M数据集需要在项目运行前进行处理，进入以下目录`run all`指定文件即可
```bash
cd data/MovieLens_1M     # 运行MovieLens_1M.ipynb
cd data/Grocery_and_Gourmet_Food    # 运行Amazon.ipynb(之后需要把映射文件和解压后的原文件拖至外层目录下)
# ！！！路径很重要，否则实验代码可能报错
```
##  LightKG复现命令
### 重要参数说明
1. 数据集与评测标准
```bash
--dataset                   # 指定数据集
--path                      # 数据集路径(需根据实际情况替换)
--test_all                  # 是否采用全量测试
--metric NDCG,HR,MRR,HIT,PRECISION   # 希望输出的指标

```
2. 自定义读取参数
```bash
--recbole_format            # 是否读取recbole数据集格式(读取原论文数据集需要)
--reader                    # 指定reader
```
3. 普通训练参数
```bash
--emb_size                  # 嵌入向量维度
--n_layers                  # GNN层数
--lr                        # 学习率
--l2                        # L2正则化系数
--mess_dropout              # 消息丢弃率
```
4. LightKG模型参数
```bash
--cos_loss                  # 是否开启对比学习
--num_neg                   # 对比训练负采样数量
--user_loss                 # 用户损失权重
--item_loss                 # 物品损失权重
```
### 运行指南

进入到指定目录下
```bash
cd src
```
1. 运行LightKG原论文数据集命令
```bash
python -u main.py   --model_name LightKG  --dataset  lastfm  --path ../LightKG_dataset    --recbole_format 1   --test_all 1   --emb_size 64   --n_layers 2   --lr 0.0005   --l2 0.00005   --mess_dropout 0.1  --cos_loss 1   --user_loss 1e-08   --item_loss 1e-07   --early_stop 20   --batch_size 2048   --epoch 200   --num_neg 10   --metric NDCG,HR,MRR,HIT,PRECISION
```

2. 运行ReChorus框架数据集命令
```bash
python -u main.py   --model_name LightKG   --dataset Grocery_and_Gourmet_Food   --reader LKGReader   --emb_size 64  --n_layers 2   --lr 0.0005   --l2 0.00005   --mess_dropout 0.1   --cos_loss 1   --user_loss 1e-08   --item_loss 1e-07  --early_stop 10   --batch_size 2048   --epoch 100   --num_neg 10   --metric NDCG,HR,MRR,HIT,PRECISION
```
### 报错处理
1. 如果出现`cuda out of memory`，这与跑代码的计算机算力有关系，代码对显存有要求（ml-1m数据集太大了），但是真的不是代码问题😭
2. 如果出现`NotImplementedError: Cannot access storage of SparseTensorImpl`，设置命令行参数--num_workers为0
3. 如果出现`AssertionError: relation overflow before graph: max=nan, n_rel=4`，删除数据集文件夹下的pkl文件重试
## 运行实验代码
进入到指定目录下
```bash
cd src
```
1. 消融实验 && 超参实验：进入`ablation_argument_draw.ipynb`文件运行
2. 对比实验
```bash
python pipeline.py
```
3. 案例分析
```bash
python case.py
```

4. 嵌入空间的语义表征可视化
```bash
python visual.py
```
## 项目核心架构

```bash
77_ReChorus_LightKG/
├── LightKG_dataset/
├── data/                         # ReChorus自带数据集
├── docs/                         # ReChorus框架文件
├── model/                        # 模型权重入口
│   ├── BPRMF
│   ├── BUIR
│   ├── LightKG                   # 我们的模型权重
│   └── ······
├── src/                       
│   ├── helpers/
│        ├── BaseReader.py
│        ├── RecBoleReader.py     # LightKG读取原数据集类
│        ├── LightKGReader.py     # LightKG读取ReChorus自带数据集类
│        ├── BaseRunner.py
│        └── LightKGRunner.py     # LightKG训练类
│   ├── log/
│   ├── models/
│        ├── BaseModel.py
│        └── general
│             └── LightKG.py      # LightKG类
│   ├── utils/
│   ├── ablation_argument_draw.ipynb
│   ├── case.py
│   ├── main.py                   # 主函数入口
│   ├── pipeline.py
│   └── visual.py
│
└── README.md
``` 

## 引用

```bash
@inproceedings{li2024rechorus2,
  title={ReChorus2. 0: A Modular and Task-Flexible Recommendation Library},
  author={Li, Jiayu and Li, Hanyu and He, Zhiyu and Ma, Weizhi and Sun, Peijie and Zhang, Min and Ma, Shaoping},
  booktitle={Proceedings of the 18th ACM Conference on Recommender Systems},
  pages={454--464},
  year={2024}
}
```