import traceback
from utils import send_request_to_api, extract_json_from_string, extract_info, query_documents, format_reference, extract_list
from evaluate import exact_match, f1_score, check_string, normalize_answer, acc_evaluate

from Keyword_Extract import extract_key_word
global RET_URL
RET_URL = "http://127.0.0.1:8061/search"
decompose_question_template = """
Please break down the following question into simpler sub-questions and respond in JSON format as shown below:
Format: {{"sub_questions": ["Sub-question 1", "Sub-question 2", ...]}}
Example:
- Question: When was the founder of craigslist born?
{{
    "sub_questions": [
        "Who was the founder of craigslist?",
        "When was him born?"
    ]
}}

User_input:
- Question: {question}
Please provide a JSON object following this format:
"""
modify_question_template = """
You are an auxiliary query assistant who modifies queries to better find answers to solve problems.
Follow these precise steps:
1. **Dependency Check**: For each sub-question, identify if it depends on the answer to any previous sub-question. 
    - State the dependency reason if it exists, otherwise, state "None".
2. **Dynamic Adjustment**: Modify the sub-question to include necessary information if a dependency is present. 
    - If no change is required, keep the original sub-question.
### Input Data:
- Key_Question:The key question that ultimately needs to be answered. The modified sub-questions should be queries that can provide crucial information for answering this question.
- Previous_QA_History: "The question-and-answer history of previous sub-questions, which provides crucial information for solving the key question and for the rewriting of subsequent sub-questions.
- Modifiable_Question: The sub-questions that need to be modified.
### Format your output as follows:
Inference_process: Dependency reason or 'None' if not dependent
Modified_question: Modified sub-question or original if no changes are required
##Example:
- Key_Question:When was the founder of craigslist born?
- Previous_QA_History:
sub_question_1:Who was the founder of craigslist?, sub_answer:Craigslist was founded by Craig Newmark.
- Modifiable_Question:"When was him born?"
Inference_process: The sub-question "When was him born?" depends on the answer to sub-question_1 because "him" refers to the previously identified founder, Craig Newmark.  
Modified_question: When was Craig Newmark born?

Now analyze the following question.Please be sure to output in the agreed format.
User input:
- Key_Question:{question}
- Previous_QA_History:{history}
- Modifiable_Question:{sub_question}
Model output:

"""

final_answer_template = """
Synthesize an answer to the original question based on the answers to sub-questions:\n\n
"Your reasoning process should be separated into two fields from the answer. In the answer field, please provide the answer as concisely as possible.The answer should be given in the form of words or phrases as much as possible. 
### Input Data:
- Original_Question:The key question that ultimately needs to be answered.
- Evidence:Question-and-answer pairs of the sub-questions split from the original question, which are used to answer the final original question.
### Format your output as follows:
Inference_process: Your reasoning process
Answer: Modified Provide answers as concisely as possible
##Output Example:
Inference_process: Based on the sub-questions and answers, I identified the series that matches the description as Animorphs, a science fantasy young adult series told in first person. The series has companion books that narrate the stories of enslaved worlds and alien species, which aligns with the nature of the companion books in the Square Enix series. 
Answer: Animorphs

Now analyze the following question.Please be sure to output in the agreed format.
User input:
- Original_Question:{question}
- Evidence:{history}
Model output:

"""

def return_json_process(text):
    res_json = extract_json_from_string(text)
    if res_json == None:
        print("不能处理的字符串： \n", res_json, '\n')
        return text
    return res_json

def EK_recaller(question, doc_list, base_num=2):
    EK_list = extract_key_word(question)
    
    
    if len(doc_list) <= base_num:
        return doc_list, EK_list

    
    result_docs = doc_list[:base_num]

    if len(EK_list) == 0:
        return result_docs, []

    
    for doc in doc_list[base_num:]:
        
        
        if all(keyword in doc['contents'] for keyword in EK_list):
            result_docs.append(doc)

    return result_docs, EK_list

def decompose_question(question):
    prompt = decompose_question_template.format(question=question)
    raw_response, tokens_count = send_request_to_api(prompt)
    send_request_to_api.add_token_count(tokens_count['total_tokens'])
    
    
    try:
        if raw_response:
            json_response = return_json_process(raw_response)
            sub_ques_list = json_response.get("sub_questions", [])
            return sub_ques_list
        else:
            return []
        
    except json.JSONDecodeError:
        print("Failed to parse JSON in response:", raw_response)
        return []



def process_sub_question(question_data, subsub_question, pre_sub_QA, chunks):
    prompt = modify_question_template.format(question=question_data, history=pre_sub_QA, sub_question=subsub_question)
    Modified_question, tokens_count = send_request_to_api(prompt)
    send_request_to_api.add_token_count(tokens_count['total_tokens'])
    sub_question = extract_info("Modified_question: ", Modified_question)
    tmp_retrieval_result = query_documents(RET_URL, sub_question, 10)
    retrieval_result, ek_list = EK_recaller(sub_question,tmp_retrieval_result,2)
    query_documents.add_token_count(len(retrieval_result))
    rel_text = format_reference(retrieval_result)
    prompt = f"Answer the following question briefly based on relevant information:\n\nQuestion: {sub_question}\n\nContext:\n{rel_text}"
    answer, tokens_count = send_request_to_api(prompt)
    send_request_to_api.add_token_count(tokens_count['total_tokens'])
    return {"sub_question": sub_question, "answer": answer, "key_words": ek_list, "relevant_chunks": retrieval_result}



def process_question(question_data, chunks, question_id, ground_truth):
    
    sub_questions = decompose_question(question_data)
    
    sub_QA_list = []
    now_sub_QA = ""
    for i, sqa in enumerate(sub_questions):
        
        now_q_a_c = process_sub_question(question_data, sqa, now_sub_QA, chunks)
        now_sub_QA += "\n" + (f"sub_question_{i + 1}: {now_q_a_c['sub_question']}, sub_answer_{i + 1}: {now_q_a_c['answer']}")
        sub_QA_list.append(now_q_a_c)

    evidence = "\n".join(
        f"sub_question_{i + 1}: {qa['sub_question']}, sub_answer: {qa['answer']}"
        for i, qa in enumerate(sub_QA_list)
    )

    
    final_prompt = final_answer_template.format(question=question_data, history=evidence)
    final_answer_text, tokens_count = send_request_to_api(final_prompt)
    send_request_to_api.add_token_count(tokens_count['total_tokens'])
    fin_ans = extract_info("Answer:", final_answer_text)
    fin_infer = extract_info("Inference_process:", final_answer_text)

    return {
        "id": question_id,
        "original_question": question_data,
        "decompose_question": sub_questions,
        "ground_truth": ground_truth,
        "final_answer": fin_ans,
        "Inference_process": fin_infer,
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
        print("详细错误信息：")
        traceback.print_exc()
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
    tag = "实验"
    use_model = "llama3.1-8B"
    method = "DEC"
    RET_URL = "http://xxx.xxx.xxx.xxx:xxx/search"
    process_file('input.json')
