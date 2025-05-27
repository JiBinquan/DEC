import argparse
from flask import Flask, request, jsonify
from flashrag.config import Config
from flashrag.utils import get_retriever

app = Flask(__name__)


parser = argparse.ArgumentParser()
parser.add_argument('--index_path', required=True, help='Path to the index file')
parser.add_argument('--corpus_path', required=True, help='Path to the corpus file')

args = parser.parse_args()


config_dict = {
    "data_dir": "path/to/data/",
    "dataset_name": "hotpotqa_20",
    "index_path": args.index_path,
    "corpus_path": args.corpus_path,
    "model2path": {
        "e5": "path/to/e5_base_v2/e5-base-v2",
        "llama3.1-8B-instruct": "/path/to/Meta-Llama-3.1-8B-Instruct"
    },
    "generator_model": "llama3.1-8B-instruct",
    "retrieval_method": "e5",
    "metrics": ["em", "f1", "acc"],
    "retrieval_topk": 10,
    "save_intermediate_data": True,
    "save_retrieval_cache": False,
}

config = Config(config_dict=config_dict)
e5_retriever = get_retriever(config)

@app.route('/search', methods=['GET'])
def search():
    query = request.args.get('query', '')
    if not query:
        return jsonify({"error": "query parameter is required"}), 400

    result = e5_retriever.search(query)
    return jsonify({"result": result})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8061)
