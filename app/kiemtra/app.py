from flask import Flask, render_template, request
# Import các thư viện cần thiết
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torch
import numpy
import re
import underthesea # Thư viện tách từ

from sklearn.model_selection import train_test_split # Thư viện chia tách dữ liệu

from transformers import AutoModel, AutoTokenizer # Thư viện BERT

import urllib.request
import re
import csv
import os
import json
import pandas as pd
import joblib
from underthesea import word_tokenize
import numpy as np

import requests

app = Flask(__name__)
# Tien xu ly

# Hàm load model BERT
def load_bert():
    v_phobert = AutoModel.from_pretrained("vinai/phobert-base")
    v_tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=False)
    return v_phobert, v_tokenizer

# Hàm chuẩn hoá câu
def standardize_data(row):
    # Xóa dấu chấm, phẩy, hỏi ở cuối câu
    row = re.sub(r"[\.,\?]+$-", "", row)
    # Xóa tất cả dấu chấm, phẩy, chấm phẩy, chấm thang, ... trong câu
    row = row.replace(",", " ").replace(".", " ") \
        .replace(";", " ").replace("“", " ") \
        .replace(":", " ").replace("”", " ") \
        .replace('"', " ").replace("'", " ") \
        .replace("!", " ").replace("?", " ") \
        .replace("-", " ").replace("?", " ")
    row = row.strip().lower()
    return row

# Hàm load danh sách các từ vô nghĩa: lắm, ạ, à, bị, vì..
def load_stopwords():
    sw = []
    with open("vietnamese-stopwords.txt", encoding='utf-8') as f:
        lines = f.readlines()
    for line in lines:
        sw.append(line.replace("\n",""))
    return sw

# Hàm tạo ra bert features
def make_bert_features(v_text):
    global phobert, sw
    v_tokenized = []
    max_len = 100 # Mỗi câu dài tối đa 100 từ
    for i_text in v_text:
        print("Đang xử lý line = ", i_text)
        # Phân thành từng từ
        line = underthesea.word_tokenize(i_text)
        # Lọc các từ vô nghĩa
        filtered_sentence = [w for w in line if not w in sw]
        # Ghép lại thành câu như cũ sau khi lọc
        line = " ".join(filtered_sentence)
        line = underthesea.word_tokenize(line, format="text")
        # print("Word segment  = ", line)
        # Tokenize bởi BERT
        line = tokenizer.encode(line)
        v_tokenized.append(line)

    # Chèn thêm số 1 vào cuối câu nếu như không đủ 100 từ
    padded = numpy.array([i + [1] * (max_len - len(i)) for i in v_tokenized])
    print('padded:', padded[0])
    print('len padded:', padded.shape)

    # Đánh dấu các từ thêm vào = 0 để không tính vào quá trình lấy features
    attention_mask = numpy.where(padded == 1, 0, 1)
    print('attention mask:', attention_mask[0])

    # Chuyển thành tensor
    padded = torch.tensor(padded).to(torch.long)
    print("Padd = ",padded.size())
    attention_mask = torch.tensor(attention_mask)

    # Lấy features dầu ra từ BERT
    with torch.no_grad():
        last_hidden_states = phobert(input_ids= padded, attention_mask=attention_mask)

    v_features = last_hidden_states[0][:, 0, :].numpy()
    print(v_features.shape)
    return v_features

sw = load_stopwords()
phobert, tokenizer = load_bert()


    

@app.route('/')
def my_form():
    return render_template('base.html')

@app.route('/', methods=['POST'])
def my_form_post():
    document = request.form['text']
    r = re.search(r"i\.(\d+)\.(\d+)", document)
    shop_id, item_id = r[1], r[2]
    ratings_url = "https://shopee.vn/api/v2/item/get_ratings?filter=0&flag=1&itemid={item_id}&limit=20&offset={offset}&shopid={shop_id}&type=0"

    offset = 0
    d = {"comment": []}
    while True:
        data = requests.get(
            ratings_url.format(shop_id=shop_id, item_id=item_id, offset=offset)
        ).json()

        # uncomment this to print all data:
        # print(json.dumps(data, indent=4))
        # leng enumerate tra ket qua duoi dang liet ke
        i = 1
        for i, rating in enumerate(data["data"]["ratings"], 1):
            d["comment"].append(rating["comment"])
            print(rating["comment"])

        if i % 20:
            break

        offset += 20
    
    df = pd.DataFrame(d)
    df=df.dropna()
    df = df.reset_index(drop=True)
    df=df[['comment']]
    df.to_csv("data_xl.csv", index=False)


    sv= open("data_xl.csv", "r", encoding='utf-8-sig')

    file_output='data_xl_label.csv'
    s=open(file_output,'w',encoding='utf-8-sig')
    s.write('text,label\n')

    model = joblib.load('save_model.pkl')
    for document in sv:
        a = document
        document01 = standardize_data(document)
        label = make_bert_features([document01])
        label = model.predict(label)
        if(label == 0):
            label = '1'
        if(label == 1):
            label = '0'
        s.write(document01)
        s.write('|')
        s.write(label)
        s.write('\n')

    good = np.count_nonzero(label)
    bad = len(label) - good

    if good>bad:
        label = "Sản phẩm này rất tốt để mua!"
    else:
        label = "Nên cẩn thẩn khi mua sản phẩm này!"

    return(render_template('base.html', variable=label))

if __name__ == "__main__":
    app.run(port='8087', threaded=False, debug=True)