#!/bin/bash

# 设置项目路径
ROOT_DIR=$(pwd)
DATA_DIR="$ROOT_DIR/data"
SRC_DIR="$ROOT_DIR/src"

python3 "$SRC_DIR/keynode_eval.py"
python3 "$SRC_DIR/activity_eval.py"
python3 "$SRC_DIR/replacement_eval.py"
python3 "$SRC_DIR/risk_eval.py"

echo "ALL ANALYSIS COMPLETED"
