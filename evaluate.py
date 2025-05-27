import argparse
import os
import re
import string
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm.rich import tqdm_rich

from utils import send_request_to_gpt,extract_json_from_string,send_request_to_api, extract_info


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def check_string(input_string):
    positive_words = ["yes", "true"]
    negative_words = ["no", "false"]

    
    input_string = input_string.lower()

    
    if any(word in input_string for word in positive_words):
        return "True"

    
    elif any(word in input_string for word in negative_words):
        return "False"

    
    return input_string


def f1_score(prediction, ground_truth):
    ZERO_METRIC = (0, 0, 0)
    prediction_tokens = prediction.split()
    ground_truth_tokens = ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return ZERO_METRIC
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall


def exact_match(prediction, ground_truth):
    
    if ground_truth in prediction:
        return 1
    return 0


LLM_EVAL_PROMPT = """
You are an experienced linguist who is responsible for evaluating the correctness of the generated responses.
You are provided with question, the generated responses and the corresponding ground truth answer.
Your task is to compare the generated responses with the ground truth responses and evaluate the correctness of the generated responses.
##Example:
Example_1:
User input:
-Question: The city where Alex Shevelev died is the capital of what region?
-Ground-truth Answer: the Lazio region
-Prediction: the answer is Lazio
Model output:
-Correctness: yes

Example_2:
User input:
-Question: Which drink is larger, the Apple-Kneel or the Flaming volcano?
-Ground-truth Answer: The flaming volcano
-Prediction: The Apple-Kneel
Model output:
-Correctness: no

Now analyze the following question.Please be sure to output in the agreed format.
User input:
-Question: {question}
-Ground-truth Answer: {answer}
-Prediction: {prediction}
Model output:

""".strip()

import json


def acc_evaluate(question: str, answer: str, prediction: str):
    prompt = LLM_EVAL_PROMPT.format(
        question=question, prediction=prediction, answer=answer
    )
    
    response, token_count = send_request_to_api(prompt)
    
    if check_string(extract_info("Correctness:", response)) == "True":
        return True
    else:
        return False



def gpt_acc_evaluate(question: str, answer: str, prediction: str):
    prompt = LLM_EVAL_PROMPT.format(
        question=question, prediction=prediction, answer=answer
    )

    

    response,token_count = send_request_to_gpt(prompt)
    

    

    if response is None:
        return False

    try:
        
        response_dict = extract_json_from_string(response)
    except json.JSONDecodeError:
        print("Error: Response is not in JSON format.")
        return False

    return response_dict.get("response") == "yes"

import json

import json


def process_json_file(input_file, output_file, log_file):
    with open(input_file, 'r') as f:
        data = json.load(f)

    total_acc = 0
    total_f1 = 0
    total_em = 0
    num_questions = len(data)

    for index, item in enumerate(data):
        now_question = item["original_question"]
        now_answer = item["ground_truth"]
        now_prediction = str(item["final_answer"])

        
        now_acc = acc_evaluate(now_question, now_answer, now_prediction)
        processed_answer = check_string(normalize_answer(now_answer))
        processed_prediction = check_string(normalize_answer(now_prediction))

        
        now_f1 = f1_score(processed_prediction, processed_answer)
        if isinstance(now_f1, tuple):
            now_f1 = now_f1[0]  

        now_em = exact_match(processed_prediction, processed_answer)

        
        item["accuracy"] = now_acc
        item["f1_score"] = now_f1
        item["exact_match"] = now_em

        
        total_acc += now_acc
        total_f1 += now_f1
        total_em += now_em

        print("now_acc ", now_acc, " now_f1 ", now_f1, " now_em ", now_em)

    
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4)

    
    avg_acc = total_acc / num_questions if num_questions else 0
    avg_f1 = total_f1 / num_questions if num_questions else 0
    avg_em = total_em / num_questions if num_questions else 0

    
    with open(log_file, 'w') as f:
        f.write(f"Total Accuracy: {avg_acc:.4f}\n")
        f.write(f"Average F1 Score: {avg_f1:.4f}\n")
        f.write(f"Average Exact Match: {avg_em:.4f}\n")

if __name__ == "__main__":

    process_json_file("path/to/input/file.json", "path/to/output/file.json", "path/to/log/file.txt")

