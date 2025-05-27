#!/bin/bash

# 1. 查询8061端口是否有正在运行的进程，如果有，杀掉
pid=$(lsof -t -i:8061)
if [ -n "$pid" ]; then
  echo "Found running process on port 8061, killing process $pid"
  kill -9 $pid
else
  echo "No process found on port 8061"
fi

# 2. 根据外部参数选择配置
index_path=""
corpus_path=""

# 判断传入的参数并选择相应配置
case "$1" in
  "wiki")
    index_path="/path/to/flashRAG/ER/corpus/wiki/data00/jiajie_jin/flashrag_indexes/wiki_dpr_100w/e5_flat_inner.index"
    corpus_path="/path/to/flashRAG/ER/corpus/wiki/wiki18_100w.jsonl"
    ;;
  "hotpotqa")
    index_path="/path/to/flashRAG/ER/corpus/hotpotqa/DV_indexes/e5_Flat.index"
    corpus_path="/path/to/flashRAG/ER/corpus/hotpotqa/hotpotqa_corpus.jsonl"
    ;;
  "2wikimqa")
    index_path="/path/to/flashRAG/ER/corpus/2wikimqa/DV_indexes/e5_Flat.index"
    corpus_path="/path/to/flashRAG/ER/corpus/2wikimqa/2wikimqa_corpus.jsonl"
    ;;
  "musique")
    index_path="/path/to/flashRAG/ER/corpus/musique/DV_indexes/e5_Flat.index"
    corpus_path="/path/to/flashRAG/ER/corpus/musique/musique_dev_ans_corpus.jsonl"
    ;;
  *)
    echo "Invalid argument! Use one of: wiki, hotpotqa, 2wikimqa, musique."
    exit 1
    ;;
esac

# 3. 配置文件日志输出
echo "Starting server with index_path: $index_path, corpus_path: $corpus_path"

# 4. 后台运行 Flask API，并将日志输出到文件
nohup python3 retriever_api.py --index_path "$index_path" --corpus_path "$corpus_path" > server.log 2>&1 &

# 输出后台运行日志
echo "Server is running in the background. Logs are being written to server.log"
