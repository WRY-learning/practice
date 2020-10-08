"""net URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""

from django.urls import path
from django.shortcuts import HttpResponse, render
import requests
import json
import os
import numpy as np
import nltk
import gensim
import time
import keras
import re
import math
from CRF import CRF
conf_words={}


entities ={
    0:"threatActor_name",
    1:"threatActor_aliases",
    2:"person",
    3:"security_team",
    4:"location",
    5:"industry",
    6:"company",
    7:"government",
    8:"target_crowd",
    9:"attack_activity",
    10:"counter_measure",
    11:"sub_activity",
    12:"IP_evil",
    13:"IP",
    14:"domain_evil",
    15:"domain",
    16:"attack_goal",
    17:"time",
    18:"tool",
    19:"function",
    20:"program_language",
    21:"sample_name",
    22:"sample_function",
    23:"string",
    24:"md5",
    25:"sha1",
    26:"sha2",
    27:"encryption_algo",
    28:"url_evil",
    29:"url",
    30:"malware",
    31:"vulnerability_cve",
    32:"vul_aliases",
    33:"protocol",
    34:"OS_name",
    35:"email_evil",
    36:"reference_word",
    37:"others"
}
nltk.data.path.append("./nltk_data")
maxSenLen=100
WORD_WIDTH=100
model = gensim.models.Word2Vec.load("word2vec_model")
conf_path="./conf_words.txt"
model_path="./lstm_5.h5"
lstm = keras.models.load_model(model_path, custom_objects={'CRF': CRF, "get_loss": CRF.get_loss})

english_sample= ['.', ',', ':', '(', ')', ';', '?', '!', '"', '\'', '[', ']', '{', '}', '`', '@', '#', '$', '%',
                      '^', '&', '*','``','\'\'','<','>','""','-','_','+','=','\\','|','/','‘','’',
                 '–','。','？','！','（','）','……', '￥','【','】','『','』','～',
                 ]

def index(request):
    return render(request, 'index.html')


def upload(request):
    value = request.body.decode('utf-8')
    value=value[value.find('=')+1:]
    res=deal_with_lstm(value)
    return HttpResponse(res)


def deal_with_lstm(value):
    sens=nltk.sent_tokenize(value)
    words=[]
    for now_sen in sens:
        words.append(nltk.word_tokenize(now_sen))
    updata_word2vec(words)
    
    new_words,n=words2array(words)
    predict=lstm_predict(n)
    res=filter(new_words,predict)
    res=format_res(res)
    return res
  
    
def format_res(res):
    res_str=""
    for now_sen in res:
        for now_word in now_sen:
            if now_word[1]==entities[37]:
                res_str=res_str+now_word[0]+" "
            else:
                res_str = res_str + now_word[0]+"\t{{"+now_word[1]+"}}\t"+" "
        res_str=res_str+"\n"
    return res_str



def regex_match(word):
        cve_pattern = "^(CVE|cve)-\d{4}-\d{4,}$"
        md5_16_pattern = "^([a-f\d]{16}|[A-F\d]{16})$"
        md5_32_pattern = "^([a-f\d]{32}|[A-F\d]{32})$"
        sha1_40_pattern = "^([a-f\d]{40}|[A-F\d]{40})$"
        sha2_64_pattern = "^([a-f\d]{64}|[A-F\d]{64})$"
        if re.match(cve_pattern, word, re.S):
            return entities[31]
        elif re.match(md5_16_pattern, word, re.S) or re.match(md5_32_pattern, word, re.S):
            return entities[24]
        elif re.match(sha1_40_pattern, word, re.S):
            return entities[25]
        elif re.match(sha2_64_pattern, word, re.S):
            return entities[26]
        else:
            return None

def filter(new_words,predict):
    def getlabel(now_word,now_predict):
        if(now_predict==38):
            return entities[37]
        elif now_word in conf_words:
            return conf_words[now_word]
        elif now_word in english_sample:
            return entities[37]
        else:
            res = regex_match(now_word)
            if res != None:
                return res
            return entities[now_predict]
        
    for i  in range(len(new_words)):
        for j in range(len(new_words[i])):
            now_words=new_words[i][j]
            now_predict=predict[i][j]
            new_words[i][j]=[now_words,getlabel(now_words,now_predict)]
    return new_words


def load_conf_word():
    global conf_words
    def check_in_entities(label):
        for now in entities:
            if entities[now] == label:
                return True
        return False
    f=open(conf_path,"r+",encoding='utf-8')
    while True:
        line=f.readline().replace("\n","").strip()
        if not line:
            break
        else:
            now_line=line.split(",")
            assert len(now_line)==2
            if check_in_entities(now_line[1])==False:
                raise  Exception('配置文件错误'+line)
            conf_words[now_line[0]]=now_line[1]
    f.close()



def updata_word2vec(words):
    model.build_vocab(words, update=True)
    model.train(words, total_examples=model.corpus_count, epochs=model.iter)
    
def split_words(words):
    num=math.ceil(len(words)/maxSenLen)
    index=0
    res=[]
    for i in range(num):
        end=0
        if index+maxSenLen<len(words):
            end=index+maxSenLen
        else:
            end=len(words)
        res.append(words[index:end])
        index=index+maxSenLen
    return res


def words2array(words):
    
    new_words=[]
    for now in words:
        new_words.extend(split_words(now))
    
    n=np.zeros((len(new_words),maxSenLen,WORD_WIDTH))
    for i in range(len(new_words)):
        for j in range(len(new_words[i])):
            n[i][j]=model[new_words[i][j]]
    
    return new_words,n





def lstm_predict(n):
    res = lstm.predict(n)
    return res


load_conf_word()

urlpatterns = [
    path('index/', index),
    path('upload/', upload),
]
