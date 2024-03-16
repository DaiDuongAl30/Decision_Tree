import pandas as pd

# Hàm đọc tập luật từ file CSV
def load_rules_from_csv(file_path):
    return pd.read_csv(file_path).to_dict(orient='records')

# Hàm dự đoán từ tập luật
def predict_from_rules(rules, instance):

    for rule in rules:
        #print(rule)
        match = True
        for key, value in rule.items():
            if key != 'Prediction':
                if key not in instance:
                    match = False
                    break
                # So sánh giá trị thuộc tính với đối tượng mới
                if pd.isna(value):
                    continue
                elif instance[key] != value:
                    match = False
                    break
        if match:
            print(rule)
            return rule['Prediction']

    return 'Unknown'

# Hàm kiểm tra đối tượng mới và dự đoán bệnh
def predict_disease(new_instance):
    # Đọc tập luật từ file CSV
    loaded_rules = load_rules_from_csv('Decision_rules.csv')

    # Dự đoán từ tập luật
    prediction = predict_from_rules(loaded_rules, new_instance)
    return prediction

# Đối tượng mới cần dự đoán
new_instance = {'Age': 'Adult', 'sex': 1, 'cp': 1, 'trest': 'Normal', 'chol': 'High Risk', 'fbs': 0, 'restecg': 0, 'thalach': 'High', 'exang': 0, 'oldpeak': 'Low', 'slope': 2, 'ca': 0, 'thal': 2}

# Dự đoán bệnh
prediction = predict_disease(new_instance)
print(f'Predicted Heart disease: {prediction}')