## LightKG based on ReChorus项目解析
by **赵景琦&廖桦淇**
###  LightKG复现环境配置

```bash
# 创建指定版本的conda环境
conda create -n LKGenv python=3.9

# 激活conda环境
conda activate LKGenv

# 安装指定版本的包！注意顺序很重要，必须先装torch
pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118

# 再装torch-scatter
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cu118.html

# 其它包，注意限制版本
pip install "numpy<2.0" "recbole==1.1.1" lightgbm xgboost ray thop
```

###  LightKG复现命令

```bash
# 激活conda环境
conda activate LKGenv

# 切换目录
cd ReChorus/src

# 运行命令，注意因为ReChorus框架是静态参数配置故命令较长
# 当前为运行 【LightKG原论文数据集】 命令
python -u main.py \
  --model_name LightKG\
  --dataset  lastfm\                  # 指定数据集
  --path /LightKG/dataset \           # 数据集路径，【请根据实际情况调整为绝对路径】
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
  --metric NDCG,HR,MRR,HIT,PRECISION \       #你希望输出的指标
  2>&1 | tee ../log/lightkg_lastfm_$(date +"%Y%m%d_%H%M%S").log   

# 当前为运行 【ReChorus框架数据集】 命令
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
  --metric NDCG,HR,MRR,HIT,PRECISION \       #你希望输出的指标
  2>&1 | tee ../log/lightkg_ml-1m_$(date +"%Y%d_%H%M%S").log
```
！！！如果出现cuda out of memory，可以把batch_size调小试试，但是真的不是代码问题😭

### 运行实验代码

```bash
# 进入到以下路径
cd ReChorus/src

# 1. 消融实验 && 超参实验
# 进入ablation_argument_draw.ipynb文件运行

# 2. 对比试验
python pipeline.py

# 3. 案例分析
python case.py

# 4. 嵌入空间的语义表征可视化
python visual.py
```
### 项目核心架构

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