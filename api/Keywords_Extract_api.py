from flask import Flask, request, jsonify
import transformers
import torch


app = Flask(__name__)


def load_pipeline(model_id):
    return transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
    )

model_id = "/path/to/flashRAG/ER/model/EK_3B_merge_200step"
pipeline = load_pipeline(model_id)


def extract_key(question_data):
    instruction = """
    Extract 1-2 keywords from the following question. The keywords should be phrases like numbers, property nouns, or proper nouns that can effectively distinguish the target document. The keywords should not have synonyms. Ensure the keywords are directly extracted from the question and provide them in a list format.
    Note that each keyword consists of only one word.
    """
    messages = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": question_data},
    ]

    prompt = pipeline.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    

    terminators = [
        pipeline.tokenizer.eos_token_id,
        pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    
    outputs = pipeline(
        prompt,
        max_new_tokens=256,
        eos_token_id=terminators,
        do_sample=True
    )

    
    prediction = outputs[0]["generated_text"][len(prompt):]
    return prediction


@app.route('/extract_key', methods=['POST'])
def handle_qa_judge():
    data = request.json
    if not all(key in data for key in ["question_data"]):
        return jsonify({"error": "Missing required parameters."}), 400

    question_data = data["question_data"]

    try:
        result = extract_key(question_data)
        return jsonify({"result": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8603)