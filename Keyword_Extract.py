
import requests
from utils import extract_list
def call_EK_api(api_url, question_data):
    payload = {
        "question_data": question_data,
    }
    try:
        response = requests.post(api_url, json=payload)
        response.raise_for_status()  
        result = response.json()
        if "result" in result:
            return result["result"]
        else:
            raise ValueError(f"Unexpected API response: {result}")
    except requests.RequestException as e:
        print(f"Error calling API: {e}")
        return None


def EK_process(question_data):
    api_url = "http://xxx.xxx.xxx.xxx:8603/extract_key"
    return call_EK_api(api_url, question_data)


def extract_key_word(question):

    output_string = EK_process(question)
    return extract_list(output_string)

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