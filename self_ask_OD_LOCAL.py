import sys
import json
import torch
import requests
from transformers import AutoTokenizer
from utils import send_request_to_api, extract_info, format_reference, query_documents
from evaluate import  exact_match, f1_score, check_string, normalize_answer, acc_evaluate


API_URL = "http://localhost:8000/generate"
RET_URL = "http://127.0.0.1:8061/search"
startup_template = """
You are a question-answering assistant designed to solve complex problems. Based on the question and given context, determine whether follow-up questions are required to resolve the original complex question. If needed, output "Follow up:<sub-question>" to ask a subsequent question. If the current knowledge suffices to provide a final answer, continue this process and output "So the final answer is: <final answer>".

Example 1:
Question: Who lived longer, Muhammad Ali or Alan Turing?
Are follow up questions needed here: Yes.
Follow up: How old was Muhammad Ali when he died?
Intermediate answer: Muhammad Ali was 74 years old when he died.
Follow up: How old was Alan Turing when he died?
Intermediate answer: Alan Turing was 41 years old when he died.
So the final answer is: Muhammad Ali 

Example 2:
Question: When was the founder of craigslist born?
Are follow up questions needed here: Yes.
Follow up: Who was the founder of craigslist?
Intermediate answer: Craigslist was founded by Craig Newmark.
Follow up: When was Craig Newmark born?
Intermediate answer: Craig Newmark was born on December 6, 1952.
So the final answer is: December 6, 1952

Your task:
Follow the same format to complete the following question-answer sequence.Note: You only need to output one line of information each time, that is, the question to be asked or the answer inferred in accordance with the agreed format. Please give it in the most concise language and don't output other irrelevant information.
There are only two types of output you can generate: "Follow up:<sub-question>" or "So the final answer is: <final answer>".

User input:

"""

answer_template = """
You are an advanced question-answering assistant. Your task is to extract relevant information from the provided documents to answer the given question. If the documents contain sufficient information to answer the question, use the information directly. 
Format your response as:
The answer is:<answer>

Example 1:
Question: What is the capital of France?
Document: The document mentions various cities in France, including Paris, Lyon, and Marseille. Paris is described as the political and cultural center of the country.
Model Output: 
The answer is: Paris

Your task:
Given the following question and document, provide your response in the same format.Please output the answer in the simplest form and don't output other irrelevant content.

Question: {question}
Document: {document}
Model Output:
"""

final_template  = """
You are an advanced question-answering assistant. Your task is to summarize the final answer to a question based on the provided history of intermediate questions and answers. Use the given history to determine and clearly state the final answer.

Format your response as:
Inference process:<Your reasoning process>
So the final answer is: <final answer>

Your task:
Given the final question and the question-answer history, determine the final answer in the same format.

Final Question: {question}
Question-Answer History: {history}
Model Output:

"""

def process_sub_question(sub_question, chunks):
    retrieval_result = query_documents(RET_URL,sub_question,3)
    
    qa_prompt = answer_template.format(question=sub_question, document=format_reference(retrieval_result))
    tmp_res, tokens_count = send_request_to_api(qa_prompt)
    send_request_to_api.add_token_count(tokens_count['total_tokens'])

    
    answer = extract_info("The answer is:", tmp_res)
    return {"sub_question": sub_question, "relevant_chunks": retrieval_result, "answer": answer}



def process_question(question_data, chunks, question_id,ground_truth, max_iter=5):
    question = question_data
    follow_ups = "Yes."
    res = ""
    early_exit = False
    sub_QA_list = []
    now_sub_QA = ""
    final_answer = ""
    Inference_process = ""
    
    for idx in range(max_iter):
        input_prompt = (
                startup_template
                + "\n"
                + f"\nQuesiton: {question}"
                + "\nAre follow up questions needed here: "
                + follow_ups
                + "\n"
                + res
        )
        gen_out, tokens_count = send_request_to_api(input_prompt)
        send_request_to_api.add_token_count(tokens_count['total_tokens'])
        if "Follow up: " in gen_out:
            new_query = extract_info("Follow up:", gen_out)
            now_q_a_c = process_sub_question(new_query, chunks)
            now_sub_QA += "\n" + (f"sub_question_{idx}: {now_q_a_c['sub_question']}, sub_answer: {now_q_a_c['answer']}")
            sub_QA_list.append(now_q_a_c)
            res += "\nFollow up: " + new_query + "\n" + "Intermediate answer:" + now_q_a_c['answer']
        if "So the final answer is: " in gen_out:
            final_answer = extract_info("So the final answer is:", gen_out)
            res += "\nSo the final answer is: " + final_answer
            Inference_process = gen_out
            early_exit = True

    if not early_exit:
        fin_prompt = final_template.format(question=question, history=res)
        tmpfin_ans, tokens_count = send_request_to_api(fin_prompt)
        send_request_to_api.add_token_count(tokens_count['total_tokens'])
        Inference_process = tmpfin_ans
        final_answer = extract_info("So the final answer is:", tmpfin_ans)
        res += "\nSo the final answer is: " + final_answer

    return {
        "id": question_id,
        "original_question": question_data,
        "ground_truth":ground_truth,
        "final_answer":final_answer,
        "Inference_process":Inference_process,
        "sub_questions": sub_QA_list
    }


import json
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed





def process_item(item):
    try:
        
        result = process_question(item['question'], item['chunks'], item['id'], item['answer'])

        
        now_question = result["original_question"]
        now_answer = result["ground_truth"]
        now_prediction = str(result["final_answer"])

        now_acc = acc_evaluate(now_question, now_answer, now_prediction)
        processed_answer = check_string(normalize_answer(now_answer))
        processed_prediction = check_string(normalize_answer(now_prediction))

        now_f1 = f1_score(processed_prediction, processed_answer)
        if isinstance(now_f1, tuple):
            now_f1 = now_f1[0]

        now_em = exact_match(processed_prediction, processed_answer)

        
        result["accuracy"] = now_acc
        result["f1_score"] = now_f1
        result["exact_match"] = now_em

        return result, now_acc, now_f1, now_em

    except Exception as e:
        print(f"Error processing item with id {item.get('id')}: {e}")
        return None, 0, 0, 0


def process_json_file(input_file, output_file, log_file, error_file, max_workers=4):
    with open(input_file, 'r') as f:
        data = json.load(f)

    total_acc = 0
    total_f1 = 0
    total_em = 0
    num_questions = len(data)
    success_num = 0

    
    with open(output_file, 'w') as f:
        f.write("[\n")
    with open(error_file, 'w') as ef:
        ef.write("[\n")

    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_item, item): i for i, item in enumerate(data)}

        for future in tqdm(as_completed(futures), total=num_questions, desc="Processing items in parallel",
                           unit="item"):
            result, acc, f1, em = future.result()
            if result is not None:
                
                total_acc += acc
                total_f1 += f1
                total_em += em
                success_num += 1

                
                with open(output_file, 'a') as f:
                    json.dump(result, f, indent=4)
                    if futures[future] < len(data) - 1:
                        f.write(",\n")
                    else:
                        f.write("\n")

                
                if not acc:  
                    with open(error_file, 'a') as ef:
                        json.dump(result, ef, indent=4)
                        ef.write(",\n")

    
    with open(output_file, 'a') as f:
        f.write("]\n")
    
    with open(error_file, 'rb+') as ef:
        ef.seek(0, 2)  
        ef.seek(ef.tell() - 2, 0)  
        ef.write(b"\n]")

    
    avg_acc = total_acc / success_num if success_num else 0
    avg_f1 = total_f1 / success_num if success_num else 0
    avg_em = total_em / success_num if success_num else 0

    gen_count, gen_time, token_count = send_request_to_api.get_statistics()
    sarch_count, sarch_time, doc_count = query_documents.get_statistics()
    
    
    avg_doc_use = doc_count / success_num if success_num else 0
    avg_sub_doc_use = doc_count / sarch_count if sarch_count else 0
    avg_token_counts = token_count / success_num if success_num else 0
    avg_sub_num = sarch_count / success_num if success_num else 0
    
    with open(log_file, 'w') as f:
        f.write(f"文件名: {log_file}\n")
        f.write(f"生成次数: {gen_count}\n")
        f.write(f"子问题个数: {avg_sub_num}\n")
        f.write(f"生成平均时间: {gen_time}\n")
        f.write(f"消耗token数: {token_count}\n")
        f.write(f"平均消耗token数: {avg_token_counts}\n")
        f.write(f"检索次数: {sarch_count}\n")
        f.write(f"检索平均时间: {sarch_time}\n")
        f.write(f"平均文档使用: {avg_doc_use}\n")
        f.write(f"子问题平均文档使用: {avg_sub_doc_use}\n")
        f.write(f"Total Success Data: {success_num}\n")
        f.write(f"Total Accuracy: {avg_acc:.4f}\n")
        f.write(f"Average F1 Score: {avg_f1:.4f}\n")
        f.write(f"Average Exact Match: {avg_em:.4f}\n")
    send_request_to_api.reset_statistics()
    query_documents.reset_statistics()


global use_model
global tag
global method
import datetime
import os
def process_file(input_json_file, max_workers=8):

    
    current_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

    
    
    


    now_filename = os.path.splitext(os.path.basename(input_json_file))[0]

    output_json_file = f"output/jsonfile/{use_model}测试输出_{method}_{tag}_{now_filename}_{current_time}.json"
    false_file = f"output/jsonfile/[仅负例]{use_model}测试输出_{method}_{tag}_{now_filename}_{current_time}.json"
    log_file = f"output/logs/{use_model}测试评估指标_{method}_{tag}_{now_filename}_{current_time}.txt"
    process_json_file(input_json_file, output_json_file, log_file, false_file, max_workers=max_workers)
    return log_file
    
    
    


import argparse
if __name__ == "__main__":
    
    '''
    parser = argparse.ArgumentParser(description="Process a file path.")
    
    parser.add_argument('file_path', type=str, help="Path to the file to process")
    
    args = parser.parse_args()
    
    process_file(args.file_path)
    '''

    tag = "补充实验"
    use_model = "Qwen2.5-14B"
    method = "Self_Ask"
    RET_URL = "http://xxx.xxx.xxx.xxx:xxx/search"
    process_file('input.json')
