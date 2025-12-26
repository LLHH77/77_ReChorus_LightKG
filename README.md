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
pip install "numpy<2.0" "recbole==1.1.1" lightgbm xgboost ray thop
```
##  数据集准备

MovieLens_1M数据集需要在项目运行前进行处理，进入以下目录`run all`指定文件即可
```bash
cd ReChorus/data/MovieLens_1M     # 运行MovieLens_1M.ipynb
```
##  LightKG复现命令

进入到指定目录下
```bash
cd ReChorus/src
```

1. 运行LightKG原论文数据集命令
```bash
# 运行命令，注意因为ReChorus框架是静态参数配置故命令较长
python -u main.py \
  --model_name LightKG\
  --dataset  lastfm\                  # 指定数据集
  --path ../LightKG_dataset \         # 数据集路径(需根据实际情况替换)
  --recbole_format 1 \                # 因为是读取原论文数据集所以需添加该参数
  --test_all 1 \                      # 全量测试,与原论文数据集训练方式一致
  --emb_size 64 \                     # 以下全为训练参数
  --n_layers 2 \
  --lr 0.0005 \
  --l2 0.00005 \
  --mess_dropout 0.1 \
  --cos_loss 1 \
  --user_loss 1e-08 \
  --item_loss 1e-07 \
  --early_stop 20 \
  --batch_size 2048 \
  --epoch 200 \
  --num_neg 10 \
  --metric NDCG,HR,MRR,HIT,PRECISION \       # 你希望输出的指标
  2>&1 | tee ../log/lightkg_lastfm_$(date +"%Y%m%d_%H%M%S").log   
```

2. 运行ReChorus框架数据集命令
```bash
python -u main.py \
  --model_name LightKG \              # 指定数据集
  --dataset MovieLens_1M \
  --reader LKGReader \                # 指定reader
  --emb_size 64 \                     # 以下全为训练参数
  --n_layers 2 \
  --lr 0.0005 \
  --l2 0.00005 \
  --mess_dropout 0.1 \
  --cos_loss 1 \
  --user_loss 1e-08 \
  --item_loss 1e-07 \
  --early_stop 10 \
  --batch_size 2048 \
  --epoch 100 \
  --num_neg 10 \
  --metric NDCG,HR,MRR,HIT,PRECISION \       # 你希望输出的指标
  2>&1 | tee ../log/lightkg_ml-1m_$(date +"%Y%d_%H%M%S").log
```
！！！如果出现cuda out of memory，可以把batch_size调小试试，但是真的不是代码问题😭
## 运行实验代码

进入到指定目录下
```bash
cd ReChorus/src
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
```
python visual.py
```
## 项目核心架构

```bash
ReChorus/
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
│   └── main.py                  # 主函数入口
│
└── ReadMe.md
``` 

```bash
LightKG/
├── LightKG.py                   # 原论文模型实现
├── main.py                      # 原论文训练入口
├── model/                       # 其它对比模型
│   ├── CFKG.py
│   ├── KGAT.py
│   ├── ······
├── yaml/                        # 原论文参数配置文件
│   ├── lastfm_LightKG.yaml
│   ├── ml-1m_LightKG.yaml
│   ├── book-crossing_LightKG.yaml
│   └── Amazon-book_LightKG.yaml
│
└── dataset/                     # 数据集目录
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