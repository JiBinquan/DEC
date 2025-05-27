#!/bin/bash

# 1. 检查8000端口是否被占用，如果被占用则kill掉相关进程
PORT=8000

# 查找占用8000端口的进程ID
PID=$(lsof -t -i:$PORT)

if [ -n "$PID" ]; then
  echo "Port $PORT is in use, killing process with PID: $PID"
  kill -9 $PID
else
  echo "Port $PORT is not in use."
fi

# 2. 设置模型路径
modelpath="/path/to/LLMs/Meta-Llama-3.1-8B-Instruct/"

# 后台执行API服务器，并将日志输出到指定文件
nohup python3 -m vllm.entrypoints.openai.api_server \
    --model $modelpath \
    --served-model-name Llama-3.1-8B \
    --max_model_len 4096 \
    --gpu_memory_utilization 0.8 \
    --trust-remote-code > log/vllm_logfile.log 2>&1 &

echo "API server started in the background. Logs will be written to log/vllm_logfile.log"
