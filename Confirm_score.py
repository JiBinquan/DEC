import json
import os
from evaluate import normalize_answer

def process_file(input_file_path):
    # 创建新的文件名，通过在原文件名加上 "_processed" 后缀
    base, ext = os.path.splitext(input_file_path)
    output_file_path = f"{base}_processed{ext}"

    with open(input_file_path, 'r') as file:
        lines = file.readlines()

    with open(output_file_path, 'w') as file:
        for i in range(len(lines)):
            # 如果当前行是 "}" 且下一行是 "{"
            if lines[i].strip() == "}":
                if i + 1 < len(lines) and lines[i + 1].strip() == "{":
                    lines[i] = "},"  # 修改为 "},"

            # 如果当前行是 "}," 且下一行是 "]"
            elif lines[i].strip() == "},":
                if i + 1 < len(lines) and lines[i + 1].strip() == "]":
                    lines[i] = "}"  # 修改为 "}"

            file.write(lines[i])  # 写回新的文件

    print(f"Processed file saved as: {output_file_path}")
    return output_file_path

# 调用函数，指定输入文件路径
input_file = 'path/to/input.json'
pos_file = process_file(input_file)


# 加载JSON文件
with open(pos_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 统计元素数量
num_elements = len(data)

# 计算平均值
total_accuracy = 0
total_f1_score = 0
total_exact_match = 0
total_cover_em = 0

for entry in data:
    if normalize_answer(entry['ground_truth']) in normalize_answer(entry['final_answer']):
        total_cover_em += 1
    total_accuracy += entry.get('accuracy', 0)
    total_f1_score += entry.get('f1_score', 0)
    total_exact_match += entry.get('exact_match', 0)

average_accuracy = total_accuracy / num_elements if num_elements > 0 else 0
average_f1_score = total_f1_score / num_elements if num_elements > 0 else 0
average_exact_match = total_exact_match / num_elements if num_elements > 0 else 0
average_cover_EM = total_cover_em / num_elements if num_elements > 0 else 0

# 输出结果
print(f"Number of elements: {num_elements}")
print(f"Average accuracy: {average_accuracy:.4f}")
print(f"Average F1 score: {average_f1_score:.4f}")
print(f"Average exact match: {average_exact_match:.4f}")
print(f"Average cover_EM: {average_cover_EM:.4f}")
