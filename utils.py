import requests
from multiprocessing import Manager, Value, Lock
import time

import time
from multiprocessing import Value, Lock

def track_function_calls(func):
    
    total_time = Value('d', 0.0)  
    call_count = Value('i', 0)  
    token_count = Value('i', 0)  
    lock = Lock()  

    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()

        
        with lock:
            total_time.value += (end_time - start_time)
            call_count.value += 1

        return result

    def get_statistics():
        
        average_time = total_time.value / call_count.value if call_count.value > 0 else 0
        return call_count.value, average_time, token_count.value

    def reset_statistics():
        with lock:
            total_time.value = 0
            call_count.value = 0
            token_count.value = 0

    def add_total_time(additional_time):
        with lock:
            total_time.value += additional_time

    def add_token_count(num):
        with lock:
            token_count.value += num

    def get_token_count():
        with lock:
            return token_count.value

    wrapper.get_statistics = get_statistics
    wrapper.reset_statistics = reset_statistics
    wrapper.add_total_time = add_total_time
    wrapper.add_token_count = add_token_count
    wrapper.get_token_count = get_token_count
    return wrapper


import requests


@track_function_calls
def send_request_to_api(prompt: str):
    url = "http://localhost:8000/v1/chat/completions"
    headers = {
        "Content-Type": "application/json"
    }

    data = {
        "model": "Llama-3.1-8B",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    }

    response = requests.post(url, headers=headers, data=json.dumps(data))

    if response.status_code == 200:
        response_data = response.json()
        message_content = response_data.get("choices", [{}])[0].get("message", {}).get("content", "No response content")

        
        usage = response_data.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        total_tokens = usage.get("total_tokens", 0)
        tokens_count = {
            "prompt_tokens":prompt_tokens,
            "completion_tokens":completion_tokens,
            "total_tokens":total_tokens
        }
        
        return message_content, tokens_count
    else:
        return f"Error: {response.status_code}, {response.text}"


@track_function_calls
def query_documents(RET_URL, query, topk=10):
    """
    发送查询请求到 FlashRAG API 并获取相关文档列表
    可以自定义返回文档的个数。
    :param query: 用户的查询问题
    :param topk: 自定义返回文档的个数，默认为 10
    :param api_url: API 服务的 URL，默认为本地的 8061 端口
    :return: 返回的相关文档列表
    """
    
    params = {'query': query}
    max_k = 10
    try:
        
        response = requests.get(RET_URL, params=params)
        
        if response.status_code == 200:
            data = response.json()
            if 'result' in data:
                result = data['result']
                
                return result[:min(topk, len(result), max_k)]  
            else:
                raise Exception("No 'result' key in response.")
        else:
            raise Exception(f"Error: Received status code {response.status_code} from API.")

    except Exception as e:
        print(f"Error querying API: {e}")
        return []


import regex
import json


def extract_json_from_string(text):
    
    pattern = regex.compile(r'\{(?:[^{}]|(?0))*\}')
    match = pattern.search(text)

    if match:
        json_str = match.group(0)
        try:
            json_data = json.loads(json_str)
            return json_data
        except json.JSONDecodeError as e:
            print(f"JSON decoding failed: {e}")
            return None
    else:
        print("No JSON found in the string.")
        return None


def extract_info(key_words, tmp_str):
        if key_words in tmp_str:
            return [l for l in tmp_str.split("\n") if key_words in l][0].split(key_words)[-1]
        else:
            return tmp_str


def format_reference(retrieval_result):
    formatted_reference = ""
    for idx, doc_item in enumerate(retrieval_result):
        content = doc_item["contents"]
        
        if "title" in doc_item:
            title = doc_item["title"]
        else:
            
            title = content.split("\n", 1)[0]
        
        text = content if "title" in doc_item else content.split("\n", 1)[1] if "\n" in content else ""
        formatted_reference += f"Context{idx+1}:\nTitle: {title}\n{text}\n"
    return formatted_reference

def split_list(lst, n):
    """将列表按每 n 个元素一组拆分"""
    return [lst[i:i+n] for i in range(0, len(lst), n)]

@track_function_calls
def num_cal():
    return 0

import re
import ast

def extract_list(input_str):
    match = re.search(r"\[.*?\]", input_str)
    if match:
        
        list_string = match.group(0)
        
        try:
            keywords_list = ast.literal_eval(list_string)
            return keywords_list
        except (ValueError, SyntaxError):
            return []