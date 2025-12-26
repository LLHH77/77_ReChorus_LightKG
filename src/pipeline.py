# -*- coding: UTF-8 -*-
import subprocess
import re
import time
import pandas as pd
import os
from datetime import datetime

# ================= 配置区域 =================
# 1. 定义数据集变量，方便后面引用
DATASET_NAME = "MovieLens_1M"  

COMMON_ARGS = [
    "--dataset", DATASET_NAME, 
    "--topk", "5,10,20,50",
    "--metrics", "HR,MRR,NDCG,PRECISION",
    "--test_all", "0",                  # 0=负采样, 1=全量。务必统一
    "--random_seed", "0",
    "--batch_size", "2048"              # 统一 Batch Size 以便对比吞吐量（可选）
]

# 模型列表
MODELS = [
    # MovieLens_1M
    ("POP", "--train 0"), 
    ("BPRMF", "--emb_size 64 --lr 1e-3 --l2 1e-6"),
    ("NeuMF", "--emb_size 64 --layers [64] --lr 5e-4 --l2 1e-7 --dropout 0.2"),
    ("LightGCN", "--emb_size 64 --n_layers 3 --lr 1e-3 --l2 1e-8"),
    ("BUIR", "--emb_size 64 --lr 1e-3 --l2 1e-6"),
    ("DirectAU", "--emb_size 64 --lr 1e-3 --l2 1e-5 --gamma 0.3 --epoch 500"),
    ("LightKG", "--emb_size 64 --n_layers 2 --lr 0.001 --l2 1e-4 --cos_loss 1 --user_loss 1e-05 --item_loss 1e-07 -mess_dropout 0.5 --reader LKGReader")
    # Grocery_and_Gourmet_Food
    # ("POP", "--train 0"), 
    # ("BPRMF", "--emb_size 64 --lr 1e-3 --l2 1e-6"),
    # ("NeuMF", "--emb_size 64 --layers [64] --lr 5e-4 --l2 1e-7 --dropout 0.2"),
    # ("CFKG", "--emb_size 64 --margin 1 --include_attr 1 --lr 1e-4 --l2 1e-8 --reader KGReader"),
    # ("LightGCN", "--emb_size 64 --n_layers 3 --lr 1e-3 --l2 1e-8"),
    # ("BUIR", "--emb_size 64 --lr 1e-3 --l2 1e-6"),
    # ("DirectAU", "--emb_size 64 --lr 1e-3 --l2 1e-5 --gamma 0.3 --epoch 500"),
    # ("LightKG", "--emb_size 64 --n_layers 2 --lr 5e-3 --l2 5e-5 --num_neg 10 --cos_loss 1 --reader LKGReader")
]

# ================= 核心解析逻辑 =================

def parse_logs(log_output):
    """
    从 Log 中提取指标和三种特定的时间
    """
    metrics = {}
    
    # 1. 提取性能指标 (Test After Training)
    pattern_metrics = r"Test After Training: \((.*?)\)"
    match = re.search(pattern_metrics, log_output)
    if match:
        items = match.group(1).split(",")
        for item in items:
            if ":" in item:
                key, val = item.split(":")
                metrics[key.strip()] = float(val)
    
    # 2. 提取时间指标
    patterns_time = {
        "Train/Epoch(s)": r"\[TIMER\] Avg Train Time: (.*?) s",  # 单轮训练时间
        "Train Total(s)": r"\[TIMER\] Total Train Time: (.*?) s",# 总训练时间
        "Inference(s)":   r"\[TIMER\] Inference Time: (.*?) s"   # 推理时间
    }
    
    for key, pattern in patterns_time.items():
        matches = re.findall(pattern, log_output)
        if matches:
            metrics[key] = float(matches[-1]) 
        else:
            metrics[key] = 0.0
            
    return metrics

def run_model(model_name, specific_args):
    print(f"\n{'='*20} Running {model_name} on {DATASET_NAME} {'='*20}") # <--- 修改：打印时显示数据集
    cmd = ["python", "main.py", "--model_name", model_name] + specific_args.split() + COMMON_ARGS
    
    start_wall = time.time()
    
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, encoding='utf-8', errors='ignore'
        )
        elapsed_wall = time.time() - start_wall
        
        # 错误检查
        if result.returncode != 0:
            print(f"Error running {model_name}:")
            print(result.stderr[-500:])
            return None
        
        # 保存详细日志
        if not os.path.exists("log"): os.makedirs("log")
        with open(f"log/{model_name}_{DATASET_NAME}_benchmark.log", "w") as f: # <--- 修改：日志文件名带上数据集
            f.write(result.stdout)
            
        # 解析日志
        metrics = parse_logs(result.stdout)
        
        if not metrics:
            print(f"Warning: No metrics found for {model_name}. Log format might be wrong.")
            return None
            
        metrics["Model"] = model_name
        metrics["Dataset"] = DATASET_NAME  # <--- 修改：在这里将数据集名称写入结果字典
        
        # 控制台简要输出
        print(f"Finished {model_name}.")
        print(f"  > Epoch Time: {metrics.get('Train/Epoch(s)', 0)}s")
        print(f"  > NDCG@20   : {metrics.get('NDCG@20', 0)}")
        
        return metrics

    except Exception as e:
        print(f"Exception: {e}")
        return None

def main():
    all_results = []
    
    # 1. 运行所有模型
    for model_name, args in MODELS:
        res = run_model(model_name, args)
        if res:
            all_results.append(res)
            
    # 2. 生成报表
    if all_results:
        df = pd.DataFrame(all_results)
        
        # 整理列顺序：模型 -> 数据集 -> 时间指标 -> 核心指标
        time_cols = ["Model", "Dataset", "Train/Epoch(s)", "Inference(s)", "Train Total(s)"]
        
        metric_cols = [c for c in df.columns if c not in time_cols]
        metric_cols.sort(key=lambda x: "NDCG@20" not in x)
        
        final_cols = time_cols + metric_cols
        # 防止有的模型没跑出来某些列报错，做个交集检查
        final_cols = [c for c in final_cols if c in df.columns]
        
        df = df[final_cols]
        
        if "NDCG@20" in df.columns:
            df = df.sort_values(by="NDCG@20", ascending=False)
        
        # 保存 CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_name = f"benchmark_{DATASET_NAME}_{timestamp}.csv" # <--- 修改：文件名也带上数据集
        df.to_csv(csv_name, index=False)
        
        print(f"\n\n{'='*20} Benchmark Completed {'='*20}")
        print(f"Results saved to: {csv_name}")
        print(df.to_string())
    else:
        print("No results collected.")

if __name__ == "__main__":
    main()