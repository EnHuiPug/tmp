## 测试

import pandas as pd
from tqdm import tqdm 
import json
import time
tqdm.pandas(desc='进度条: ')

import requests
from requests.adapters import HTTPAdapter
from urllib3 import Retry
def requests_retry_session(retries=3, backoff_factor=1, session=None):
    session = session or requests.Session()
    retry = Retry(total=retries, read=retries, connect=retries, backoff_factor=backoff_factor)
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session

def get_request(url, **kwargs):
    with requests_retry_session() as session:
        r = session.get(url, **kwargs)
    r.raise_for_status()
    return r


def post_request(url, **kwargs):
    with requests_retry_session() as session:
        r = session.post(url, **kwargs)
    r.raise_for_status()
    return r



def test_classify(content):
    URL = "http://localhost:8015/cpmbee/testclassify"
    addi = {"role":"user", "content":"现在你是一个文本分类器，“{}”是[知识图谱回复, 投顾问答, 业务功能]里的什么类别？"}
    addi["content"] = addi["content"].format(content)
    pre.append(addi)

    data = {
            "messages": pre,
            "temperature": 0.01,
            "user_id": "xusheng"
           }
    headers = {"Content-Type": "application/json"}
    try:
        import time
        time.sleep(1.2)
        current_response = post_request(url=URL, data=json.dumps(data), headers=headers, timeout=30)
        tmp = current_response.json()
        print(tmp)
        return tmp["result"]
    except requests.exceptions.RequestException as e:
        print(e)


def test_ner(content):
    URL = "http://localhost:8015/cpmbee/testner"
    addi = {"role":"user", "content":"现在你是一个文本分类器，“{}”是[知识图谱回复, 投顾问答, 业务功能]里的什么类别？"}
    addi["content"] = addi["content"].format(content)
    pre.append(addi)

    data = {
            "messages": pre,
            "temperature": 0.01,
            "user_id": "xusheng"
           }
    headers = {"Content-Type": "application/json"}
    try:
        import time
        time.sleep(1.2)
        current_response = post_request(url=URL, data=json.dumps(data), headers=headers, timeout=30)
        tmp = current_response.json()
        print(tmp)
        return tmp["result"]
    except requests.exceptions.RequestException as e:
        print(e)




##### 测试分类

data = pd.read_csv("testdata.csv")

iter_times = []
reslist = []
for index,row in data.iterrows():
    stime = time.time()
    res = test_classify(row["question"])
    iter_times.append(time.time()-stime)
    reslist.append(res)
data["infer"]  = reslist
acc = round(sum(data["infer"]==data["分类"])/200, 4)
max_time = round(max(iter_times), 4)
min_time = round(min(iter_times), 4)
avg_time = round(sum(iter_times)/200, 4)
print("分类测试结果：")
print("分类准确率：{}".format(acc))
print("最长请求时间：{}".format(max_time))
print("最短请求时间：{}".format(min_time))
print("平均请求时间：{}".format(avg_time))


########### 测试ner
with open("nertest.json") as f:
    ner_data = json.load(f)

iter_times = []
reslist = []
for key in ner_data.keys():
    stime = time.time()
    res = test_ner(key)
    iter_times.append(time.time()-stime)
    if res==ner_data[key]:
        reslist.append(1)
    else:
        reslist.append(0)
        
acc = round(sum(reslist)/200, 4)
max_time = round(max(iter_times), 4)
min_time = round(min(iter_times), 4)
avg_time = round(sum(iter_times)/200, 4)
print("ner测试结果：")
print("ner准确率：{}".format(acc))
print("最长请求时间：{}".format(max_time))
print("最短请求时间：{}".format(min_time))
print("平均请求时间：{}".format(avg_time))











