import math
import pandas as pd
from collections import Counter

# Đọc dữ liệu từ file CSV sử dụng pandas
def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

# Hàm tính độ tinh khiết của một tập dữ liệu
def calculate_entropy(data):
    labels = data['target']
    label_counts = Counter(labels)
    entropy = 0.0
    total_instances = len(labels)

    for count in label_counts.values():
        probability = count / total_instances
        entropy -= probability * (probability and math.log2(probability))

    return entropy

# Hàm chia dữ liệu thành các tập con dựa trên giá trị của thuộc tính
def partition(data, attribute):
    partitions = {}
    for index, row in data.iterrows():
        value = row[attribute]
        if value not in partitions:
            partitions[value] = []
        partitions[value].append(row)
    return partitions


# Hàm tìm thuộc tính tốt nhất để chia dữ liệu
def find_best_attribute(data, attributes):
    max_info_gain = float('-inf') #đảm bảo best_attribute sau đó được gán với giá trị khác None
    best_attribute = None
    current_entropy = calculate_entropy(data)#entropy(S)

    for attribute in attributes:
        attribute_entropy = 0.0
        partitions = partition(data, attribute)

        for partition_data in partitions.values():
            partition_entropy = calculate_entropy(pd.DataFrame(partition_data))
            attribute_entropy += (len(partition_data) / len(data)) * partition_entropy

        info_gain = current_entropy - attribute_entropy

        if info_gain > max_info_gain:
            max_info_gain = info_gain
            best_attribute = attribute

    return best_attribute

# Hàm xây dựng cây quyết định
def build_decision_tree(data, attributes):
    labels = data['target']

    if len(set(labels)) == 1:
        return labels.iloc[0] #trả về dữ liệu tương ứng với chỉ số 0 trong series

    if not attributes:
        return Counter(labels).most_common(1)[0][0] #[(a, b), (), ..., ()]: trả về giá trị đầu tiên của tuple đầu tiên


    best_attribute = find_best_attribute(data, attributes)  #Tìm ra thuộc tính tốt nhất
    tree = {best_attribute: {}} #Tạo cây dươi dạng từ điển: với key là gốc của cây còn value là phần còn lại của cây được lưu trữ dưới dạng một từ điển
    remaining_attributes = [attr for attr in attributes if attr != best_attribute]
    partitions = partition(data, best_attribute)

    for value, partition_data in partitions.items():
        tree[best_attribute][value] = build_decision_tree(pd.DataFrame(partition_data), remaining_attributes)
    return tree

# Hàm sinh ra tập luật từ cây quyết định
def generate_rules(tree, rule, rules):
    attribute = list(tree.keys())[0]#hàm lấy gốc
    sub_tree = tree[attribute]#hàm lấy phần còn lại của cây trừ gốc
    for value, sub_data in sub_tree.items():
        new_rule = rule.copy() # tình huống 1 gốc có nhiều nhánh rẽ ra
        new_rule[attribute] = value #tạo liên kết trực tiếp giữa cây mẹ và cây con thông qua nhánh

        if isinstance(sub_data, dict):
            generate_rules(sub_data, new_rule, rules)
        else:#điều kiện dừng
            new_rule['Prediction'] = sub_data
            rules.append(new_rule)

# Hàm lưu trữ tập luật vào file CSV
def save_rules_to_csv(rules, file_path):
    df = pd.DataFrame(rules)
    # Di chuyển cột 'Prediction' về cuối cùng của DataFrame
    columns = df.columns.tolist()
    columns.remove('Prediction')
    columns.append('Prediction')
    df = df[columns]
    df.to_csv(file_path, index=False)

# Đường dẫn tới file dữ liệu
file_path = 'Heart4.csv' #file_path này là của hàm trên cùng

# Đọc dữ liệu
data = load_data(file_path)

# Loại bỏ cột nhãn 'target' khỏi danh sách thuộc tính
attributes = list(data.columns)
attributes.remove('target')

# Xây dựng cây quyết định từ dữ liệu huấn luyện
tree = build_decision_tree(data, attributes)
#print(tree)
# Sinh ra tập luật từ cây quyết định
rules = []
generate_rules(tree, {}, rules)

# Lưu trữ tập luật vào file CSV
save_rules_to_csv(rules, 'decision_rules_2.csv')
