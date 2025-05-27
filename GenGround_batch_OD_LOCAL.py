import sys
import traceback

from utils import send_request_to_api, extract_info, format_reference, query_documents, \
    split_list
from evaluate import exact_match, f1_score, check_string, normalize_answer, acc_evaluate


API_URL = "http://localhost:8000/generate"
global RET_URL
RET_URL = "http://127.0.0.1:8061/search"
startup_template = """
"Please decompose the following multi-hop question into sub-questions, then solve it step by step using a deductive approach. You will alternate between two actions: 'Deduce' and 'Answer'. 
- 'Deduce': Analyze the current context, identify the key aspects of the question, and formulate a relevant sub-question that leads toward answering the overall question.
- 'Answer': Provide a clear and concise answer to the deduced sub-question.
Repeat this process of 'Deduce' and 'Answer' until you have addressed all sub-questions, and derive the final answer to the original question.Note that you only need to generate a pair of "Deduce" and "Answer" each time.

Example_1:
- Question: What is the capital of France, and how far is it from London?
  - Deduce: What is the capital of France?
  - Answer: The capital of France is Paris.
  - Deduce: How far is Paris from London?
  - Answer: The distance between Paris and London is approximately 344 kilometers.
Example_2:
- Question: If John has 10 apples and gives 4 to Jane, how many apples does he have left, and how many does Jane have now?
  - Deduce: How many apples does John have left after giving 4 to Jane?'
  - Answer: John has 6 apples left.
  - Deduce: How many apples does Jane have now?'
  - Answer: Jane has 4 apples."

User input:
- Question:{question}
- History:{history}
Model output：
- Deduce: 
- Answer: 
"""

answer_template = """
You are an intelligent question-answering assistant. Your task is to receive a given question, answer, and a list of related documents. Use the evidence from the document list to revise the answer. You should encapsulate the evidence using "<ref></ref>" and mark the revised answer using "<revise> </revise>".If you can't find the evidence to revise the answer, just output <ref>Empty<ref>.

Here are some examples:
Example_1:
Question: What is the name of the annual documentary film festival presented by the fortnightly published British journal of literary essays?
Original Answer: The Fortnightly Review Documentary Film Festival.
Documents:
...
The annual documentary film festival presented by the fortnightly published British journal of literary essays is called the London International Documentary Festival (LIDF)
...

Model output:
<ref> The annual documentary film festival presented by the fortnightly published British journal of literary essays is called the London International Documentary Festival (LIDF)</ref>.
<revise>the London International Documentary Festival (LIDF)</revise>.

Example_2:
Question: Who invented the telephone?
Original Answer: Alexander Graham Bell
Documents:
...
The document does not mention the inventor of the telephone.
...

Model output:
<ref>Empty<ref>


User input:
Question:{question}

Original Answer:{initial_answer}

Documents:
{documents}

Now, provide your revised answer and the related evidence following the specified format.
Model output:
"""

final_template = """
You are an advanced question-answering assistant. Your task is to summarize the final answer to a question based on the provided history of intermediate questions and answers. Use the given history to determine and clearly state the final answer.

Format your response as:
Inference process:<Your reasoning process>
So the final answer is: <final answer>

Your task:
Given the final question and the question-answer history, determine the final answer in the same format.Please output the answers as briefly as possible. For example:yes,no,Rage Against the Machine

Final Question: {question}
Question-Answer History: {history}
Model Output:

"""
index_path = ""
corpus_path = ""



def process_sub_question(sub_question, tmp_ans, chunks):
    retrieval_result = query_documents(RET_URL, sub_question, 9)
    ret_list = split_list(retrieval_result, 3)
    answer = tmp_ans
    
    for tmp_ret in ret_list:
        query_documents.add_token_count(3)
        prompt = answer_template.format(question=sub_question, initial_answer=tmp_ans,
                                        documents=format_reference(tmp_ret))
        now_res, tokens_count = send_request_to_api(prompt)
        send_request_to_api.add_token_count(tokens_count['total_tokens'])
        if "<ref>Empty<ref>" in now_res:
            continue
        else:
            answer = now_res
            break
    return {"sub_question": sub_question, "relevant_chunks": retrieval_result, "answer": answer}



def process_question(question_data, chunks, question_id, ground_truth, max_iter=5):
    question = question_data
    his = ""
    sub_QA_list = []
    now_sub_QA = ""
    final_answer = ""
    Inference_process = ""
    
    for idx in range(max_iter):
        input_prompt = startup_template.format(question=question, history=his)
        gen_out, tokens_count = send_request_to_api(input_prompt)
        send_request_to_api.add_token_count(tokens_count['total_tokens'])
        if "Deduce" in gen_out:
            new_query = extract_info("Deduce:", gen_out)
            new_ans = extract_info("Answer:", gen_out)
            now_q_a_c = process_sub_question(new_query, new_ans, chunks)
            sub_QA_list.append(now_q_a_c)
            his += "\n  - Deduce: " + new_query + "\n" + "  - Answer: " + now_q_a_c['answer']
        else:
            continue

    fin_prompt = final_template.format(question=question, history=his)
    tmpfin_ans, tokens_count = send_request_to_api(fin_prompt)
    send_request_to_api.add_token_count(tokens_count['total_tokens'])
    Inference_process = tmpfin_ans
    final_answer = extract_info("So the final answer is:", tmpfin_ans)
    his += "\nSo the final answer is: " + final_answer

    return {
        "id": question_id,
        "original_question": question_data,
        "ground_truth": ground_truth,
        "final_answer": final_answer,
        "Inference_process": Inference_process,
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
    tag = "补充实验"
    use_model = "Qwen2.5-14B"
    method = "GenGround"
    RET_URL = "http://xxx.xxx.xxx.xxx:xxx/search"
    process_file('input.json')
