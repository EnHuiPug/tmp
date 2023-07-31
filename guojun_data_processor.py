import re
import pandas as pd
import random
import json
from tqdm import tqdm
import os


def clear_data(data: str):
    """
    清洗数据：url, 多个\n, markdown标记
    """
    data = data.replace("<n>", "\n")
    data = data.replace("\u3000", "")
    data = data.replace("\u200b", "")
    data = data.replace('&quot', '"').replace('&amp', '&').replace('&lt', '<').replace('&gt', '>').replace('&nbsp',' ').replace('&#8226', ' ').replace("<", "<<")
    data = re.sub(' +', '\n', data)
    data = re.sub('\r\n+', '\n', data)
    data = re.sub('\n;+', '\n', data)
    data = re.sub('\n+', '\n', data)

    return data

def process_ner_data(file_name: str):
    """
    识别文中包含的股票、基金、指标、人名、板块
    原始数据：{'input': '分析一下东珠生态的短期走势,会不会出现回调?', '问题类别': '股票分析', '实体': [{'实体类别': '股票', '实体名': ['东珠生态']}]}
    """
    prompt_list = ["识别文中包含的股票名、基金名、指标、人名、板块"]
    num = len(prompt_list) - 1

    res_list = []
    with open(file_name,'r', encoding='utf8') as rf:
        for line in rf.readlines():
            line = line.strip().replace('\'','"')
            json_obj = json.loads(line)

            res = {}
            res['input'] = clear_data(json_obj['input'])
            res['prompt'] = prompt_list[random.randint(0, num)]
            ans = {"股票":"","基金":"","指标":"","人名":"","板块":""}
            for entity in json_obj['实体']:
                entity_type = entity['实体类别']
                ans[entity_type] = "<sep>".join(entity['实体名'])
            res['<ans>'] = ans
            res_list.append(res)
    return res_list

def process_query_classify_data(file_name,type_name,num_type_dict,type_num_dict):
    """
    识别问题的意图
    原始数据: {'input': '分析一下东珠生态的短期走势,会不会出现回调?', '问题类别': '股票分析', '实体': [{'实体类别': '股票', '实体名': ['东珠生态']}]}
    """
    prompt_list = ["识别问题的意图"]
    num = len(prompt_list) - 1

    res_list = []
    with open(file_name,'r', encoding='utf8') as rf:
        for line in rf.readlines():
            line = line.strip().replace('\'','"')
            json_obj = json.loads(line)
            res = {}
            res['input'] = clear_data(json_obj['input'])
            res['prompt'] = prompt_list[random.randint(0, num)]
            res['options'] = num_type_dict
            res['<ans>'] = type_num_dict[type_name]
            res_list.append(res)
    return res_list

def write_json(res_list: list, file_name: str):
    """
    写文件，每行一个json
    """
    with open(file_name, 'w', encoding='utf8') as wf:
        for res in res_list:
            json.dump(res, wf, ensure_ascii=False)
            wf.write('\n')

def create_ner_train_data(folder_path:str):
    """
    构造ner训练数据
    """
    res_lists = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            res_list = process_ner_data(file_path)
            res_lists.extend(res_list)
    return res_lists
    
    
def create_query_classify_train_data(folder_path:str):
    """
    构造问题分类训练数据
    """
    num_type_dict = {}
    type_num_dict = {}

    i = 0
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            type_name = file_path.replace(folder_path,"").replace(".txt","")
            num = '<option_'+str(i)+'>'
            i = i+1
            num_type_dict[num] = type_name
            type_num_dict[type_name] = num    

    res_lists = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            type_name = file_path.replace(folder_path,"").replace(".txt","")
            res_list = process_query_classify_data(file_path,type_name,num_type_dict,type_num_dict)
            res_lists.extend(res_list)
    return res_lists
    

if __name__ == '__main__':
    folder_path = '../data/guojun/txt_data/'  # 文件夹路径，需要替换为实际路径

    # 构造命名实体识别训练数据
    res_lists = create_ner_train_data(folder_path)
    save_file_name = os.path.join(folder_path,"../json_data/guojun-ner.json")
    write_json(res_lists, save_file_name)

    # 构造问题分类训练数据
    res_lists = create_query_classify_train_data(folder_path)
    save_file_name = os.path.join(folder_path,"../json_data/guojun-query-classify.json" )
    write_json(res_lists, save_file_name)