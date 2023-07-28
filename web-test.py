from flask import Flask, request, jsonify
import threading
import torch
from cpm_live.generation.bee import CPMBeeBeamSearch
from cpm_live.models import CPMBeeTorch, CPMBeeConfig
from cpm_live.tokenizers import CPMBeeTokenizer
from flask_cors import CORS
from questionWrap import Wrapper
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '1'

app = Flask(__name__)
CORS(app)

# 加载模型
config = CPMBeeConfig.from_json_file("config/cpm-bee-10b.json")
ckpt_path = "/mnt/xusheng/modelbest/cpm-bee-10b/pytorch_model.bin"
# config = CPMBeeConfig.from_json_file("config/cpm-bee-20b.json")
# ckpt_path = "/mnt/20b.pt"
tokenizer = CPMBeeTokenizer()
model = CPMBeeTorch(config=config)
model.load_state_dict(torch.load(ckpt_path))
import bminf
with torch.cuda.device(0):
    model = bminf.wrapper(model, quantization=False)
model.half().cuda()
beam_search = CPMBeeBeamSearch(
    model=model,
    tokenizer=tokenizer,
)


# 创建线程锁和计数器
lock = threading.Lock()
counter = 0
MAX_CONCURRENT_REQUESTS = 5  # 最大并发请求数

options = {"<option_0>": "板块查询", "<option_1>": "板块分析", "<option_2>": "板块筛选", "<option_3>": "大盘分析", "<option_4>": "股票查询", "<option_5>": "股票分析", "<option_6>": "基金查询", "<option_7>": "基金分析", "<option_8>": "资产配置", "<option_9>": "资讯查询"}

@app.route('/cpmbee/completion', methods=['POST'])
def test():
        #     response={
        #     "questionType":"股票分析", 
        #     "entity":[
        #         {"entityType": "股票", "entityName":["贵州茅台"]}, 
        #         {"entityType": "基金", "entityName":["贵州茅台"]}, 
        #         {"entityType": "指标", "entityName":["贵州", "茅台"]}, 
        #         {"entityType": "人名", "entityName":[""]}
        #     ],
        #     "relatedMatching":"XXXX" ,
        #     "bottomLineReply":"XXXXX"
        # }
    global counter

    # 请求过载，返回提示信息
    if counter >= MAX_CONCURRENT_REQUESTS:
        return jsonify({'message': '请稍等再试'})

    # 获取线程锁
    with lock:
        counter += 1
    result = {}
    try:
        input_seq = request.json
        max_length = input_seq.get('max_length', 500)
        repetition_penalty = input_seq.get('repetition_penalty', 1.1)
        temperature = input_seq.get('temperature', 0.7)
        question = input_seq.get('question')
        cur_wrapper = Wrapper(question)
        res={
            "questionType":"", 
            "entity":[],
            "relatedMatching":[] ,
            "bottomLineReply":""
        }

        inputdatalist = []
        inputdatalist.append({"question": cur_wrapper.quesionType_prompt, "<ans>": ""})
        # inputdatalist.append({"question": cur_wrapper.entity_prompt, "<ans>": ""})
        # inputdatalist.append({"question": cur_wrapper.relatedMatching_prompt, "<ans>": ""})
        inputdatalist.append({"question": cur_wrapper.bottomLineReply_prompt, "<ans>": ""})
            
        response = beam_search.generate(inputdatalist, max_length=max_length, repetition_penalty=repetition_penalty)
        res["questionType"] = response[0]["<ans>"]
        # res["entity"] = [response[1]["<ans>"]]
        # res["relatedMatching"] = [response[2]["<ans>"]]
        res["bottomLineReply"] = response[1]["<ans>"]

        result['status'] = "SUCCESS"
        result['response'] = res
        return jsonify(result)
    finally:
        # 释放线程锁并减少计数器
        with lock:
            counter -= 1


@app.route('/cpmbee/chat', methods=['POST'])
def chat():
       
    global counter

    # 请求过载，返回提示信息
    if counter >= MAX_CONCURRENT_REQUESTS:
        return jsonify({'message': '请稍等再试'})

    # 获取线程锁
    with lock:
        counter += 1
    result = {}
    try:
        input_seq = request.json
        max_length = input_seq.get('max_length', 200)
        repetition_penalty = input_seq.get('repetition_penalty', 1.1)
        temperature = input_seq.get('temperature', 0.7)
        question = input_seq.get('question')
        # cur_wrapper = Wrapper(question)
        res={
            "answer":"", 
        }

        inputdatalist = []
        inputdatalist.append({"question": question, "<ans>": ""})
        
            
        response = beam_search.generate(inputdatalist, max_length=max_length, repetition_penalty=repetition_penalty)
        res["answer"] = response[0]["<ans>"]
       
        result['status'] = "SUCCESS"
        result['response'] = res
        return jsonify(result)
    finally:
        # 释放线程锁并减少计数器
        with lock:
            counter -= 1

@app.route('/cpmbee/test', methods=['POST'])
def testclassify():
       
    global counter

    # 请求过载，返回提示信息
    if counter >= MAX_CONCURRENT_REQUESTS:
        return jsonify({'message': '请稍等再试'})

    # 获取线程锁
    with lock:
        counter += 1
    result = {}
    try:
        input_seq = request.json
        max_length = input_seq.get('max_length', 500)
        repetition_penalty = input_seq.get('repetition_penalty', 1.1)
        temperature = input_seq.get('temperature', 0.7)
        question = input_seq.get('question')
        # cur_wrapper = Wrapper(question)
        res={
            "questionType":"", 
            "entity":"", 
        }

        inputdatalist = []
        inputdatalist.append({"input": question, "prompt": "识别问题的意图",  "options": options, "<ans>": ""})
            
        response = beam_search.generate(inputdatalist, max_length=max_length, repetition_penalty=repetition_penalty)
        if response[0]["<ans>"] and response[0]["<ans>"] in options:
            res["questionType"] = options[response[0]["<ans>"]]
        
        result['status'] = "SUCCESS"
        result['response'] = res
        return jsonify(result)
    finally:
        # 释放线程锁并减少计数器
        with lock:
            counter -= 1




@app.route('/cpmbee/testner', methods=['POST'])
def testner():
       
    global counter

    # 请求过载，返回提示信息
    if counter >= MAX_CONCURRENT_REQUESTS:
        return jsonify({'message': '请稍等再试'})

    # 获取线程锁
    with lock:
        counter += 1
    result = {}
    try:
        input_seq = request.json
        max_length = input_seq.get('max_length', 500)
        repetition_penalty = input_seq.get('repetition_penalty', 1.1)
        temperature = input_seq.get('temperature', 0.7)
        question = input_seq.get('question')
        # cur_wrapper = Wrapper(question)
        res={
            "questionType":"", 
            "entity":"", 
        }

        inputdatalist = []
        inputdatalist.append({"input": question, "prompt": "识别文中包含的股票名、基金名、指标、人名、板块，并以json输出。", "<ans>": ""})
        
            
        response = beam_search.generate(inputdatalist, max_length=max_length, repetition_penalty=repetition_penalty)
        if response[0]["<ans>"]:
            res["entity"] = options[response[0]["<ans>"]]
       
        result['status'] = "SUCCESS"
        result['response'] = res
        return jsonify(result)
    finally:
        # 释放线程锁并减少计数器
        with lock:
            counter -= 1


if __name__ == '__main__':
    print("Flask 服务器已启动")
    app.run(host='0.0.0.0', port=8015)