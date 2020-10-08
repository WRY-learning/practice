import nltk
from nltk import data
import os
import numpy as np
from tensorflow.keras.models import Sequential
import tensorflow as tf
from CRF import CRF
import gc
import sys
path="./val_datas"
tagSum=39
maxSenLen=100
BERT_WIDTH=100
from keras import backend as K
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
"""
对列表中的模型进行验证，找出最优的模型（泛化能力）
"""


model_name=["lstm_1.h5",
            "lstm_2.h5",
            "lstm_3.h5",
            "lstm_4.h5",
            "lstm_5.h5",
            "lstm_6.h5",
            "lstm_7.h5",
            "lstm_8.h5",
            "lstm_9.h5",
            "lstm_10.h5",
            ]
WORD_WIDTH=100
def readOneFile(file):
    file_path=os.path.join(path,file)
    f = open(file_path, "r+", encoding='utf-8')
    sens=[]
    label=[]
    while True:
        line=f.readline().replace("\n","")
        if not line:
            break
        else:
            tmp=line.split("\t\t")
            if len(tmp[1].split(" "))<=maxSenLen:
                sens.append(tmp[0])
                label.append(tmp[1])
            else:
                pass
    return sens,label


def changeLabels(labels):
    def getEntityIndex(target):
        for i in entities:
            if entities[i] == target:
                return i

    res=[]
    for line in labels:
        nums=[]
        tmp=line.split(" ")
        for now_entity in tmp:
            nums.append(getEntityIndex(now_entity))

        res.append(nums)

    final_np=np.full((len(res),maxSenLen),fill_value=tagSum-1,dtype=int)

    for i in range(len(res)):
        for j in range(len(res[i])):
            final_np[i][j]=res[i][j]

    return final_np

def changeSentenses(sens):
    import gensim
    model = gensim.models.Word2Vec.load("word2vec_model")
    n = np.zeros((len(sens), maxSenLen, WORD_WIDTH), dtype=float)
    for i in range(len(sens)):
        words = nltk.tokenize.word_tokenize(sens[i])
        for j in range(len(words)):
            n[i][j] = model[words[j]]

    return n


def readFiles():
    global train_x
    global train_Y
    data.path.append("./nltk_data")
    files = os.listdir(path)
    sentenses=[]
    labels=[]
    for file in files:
        tmp1,tmp2=readOneFile(file)
        sentenses.extend(tmp1)
        labels.extend(tmp2)

    train_Y=changeLabels(labels).reshape((-1,maxSenLen,1))
    train_x=changeSentenses(sentenses)





if __name__=="__main__":
    readFiles()
    print(train_x.shape)
    for now in model_name:
        model = tf.keras.models. \
            load_model("./"+now, custom_objects={'CRF': CRF, "get_loss": CRF.get_loss})
        model.compile('adam', loss={'crf_layer': model.layers[5].get_loss})
        scores = model.evaluate(train_x, train_Y, verbose=0)
        print(now+"   "+str(scores))
        gc.collect()
        K.clear_session()