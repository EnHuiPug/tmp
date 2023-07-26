from flask import Flask, request, jsonify
import threading
import torch
from cpm_live.generation.bee import CPMBeeBeamSearch
from cpm_live.models import CPMBeeTorch, CPMBeeConfig
from cpm_live.tokenizers import CPMBeeTokenizer
from opendelta import LoraModel
from flask_cors import CORS
from questionWrap import Wrapper
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '1'

app = Flask(__name__)
CORS(app)

# 加载模型
config = CPMBeeConfig.from_json_file("config/cpm-bee-10b.json")
ckpt_path = "/mnt/xusheng/modelbest/cpm-bee-10b/pytorch_model.bin"
tokenizer = CPMBeeTokenizer()
model = CPMBeeTorch(config=config)
model.load_state_dict(torch.load(ckpt_path))
model.half().cuda()
beam_search = CPMBeeBeamSearch(
    model=model,
    tokenizer=tokenizer,
)


# 创建线程锁和计数器
lock = threading.Lock()
counter = 0
MAX_CONCURRENT_REQUESTS = 5  # 最大并发请求数


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
        max_length = input_seq.get('max_length', 200)
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
        inputdatalist.append({"question": cur_wrapper.entity_prompt, "<ans>": ""})
        inputdatalist.append({"question": cur_wrapper.relatedMatching_prompt, "<ans>": ""})
        inputdatalist.append({"question": cur_wrapper.bottomLineReply_prompt, "<ans>": ""})
            
        response = beam_search.generate(inputdatalist, max_length=max_length, repetition_penalty=repetition_penalty)
        res["questionType"] = response["response"][0]["<ans>"]
        res["entity"] = [response["response"][1]["<ans>"]]
        res["relatedMatching"] = [response["response"][2]["<ans>"]]
        res["bottomLineReply"] = response["response"][3]["<ans>"]

        result['status'] = "SUCCESS"
        result['response'] = response
        return jsonify(result)
    finally:
        # 释放线程锁并减少计数器
        with lock:
            counter -= 1


if __name__ == '__main__':
    print("Flask 服务器已启动")
    app.run(host='0.0.0.0', port=8015)