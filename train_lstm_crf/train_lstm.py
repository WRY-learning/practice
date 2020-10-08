import nltk
from nltk import data
import os
import numpy as np
import gensim
from tensorflow.keras.models import Sequential
import tensorflow as tf
from CRF import CRF
import gc
import sys
model_path = 'word2vec_model'
path="./datas"
tagSum=39#类别总数
maxSenLen=100#最大句子长度
WORD_WIDTH=100#一个词向量大小
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


train_x=None#数据
train_Y=None#标签
from keras import backend as K


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

def readFiles():
    global train_x
    global train_Y
    data.path.append("./nltk_data")#加载nltk文件
    files = os.listdir(path)
    sentenses=[]#保存句子
    labels=[]#保存标签
    for file in files:
        tmp1,tmp2=readOneFile(file)
        sentenses.extend(tmp1)
        labels.extend(tmp2)
    train_Y = changeLabels(labels).reshape((-1, maxSenLen, 1))
    train_x = sentenses


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
"""
由于不能将全部的数据都装入内存，所以使用生成器
batch_size批处理大小
"""
def generate_imp(batch_size):
    global train_x
    global train_Y
    total_len = len(train_x)
    start = 0
    model = gensim.models.Word2Vec.load("word2vec_model")
    while 1:
        end = 0
        if start + batch_size > total_len:
            end = total_len
        else:
            end = start + batch_size


        sens=train_x[start:end]
        n=np.zeros((len(sens),maxSenLen,WORD_WIDTH),dtype=float)
        for i in range(len(sens)):
            words=nltk.tokenize.word_tokenize(sens[i])
            for j in range(len(words)):
                n[i][j]=model[words[j]]
        yield (n, train_Y[start:end][:][:])
        start = end
        if start == total_len:
            start = 0
        """下面两句用于内存空间的释放！重要！"""
        gc.collect()
        K.clear_session()


if __name__=="__main__":
    readFiles()

    """模型的搭建"""
    myModel = Sequential()#创建序列
    myModel.add(tf.keras.layers.Input(shape=(maxSenLen, WORD_WIDTH)))#输入层
    myModel.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(
        tagSum, return_sequences=True)))#BiLSTM
    myModel.add(tf.keras.layers.Dropout(0.3))#防止过拟合
    myModel.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(
        tagSum, return_sequences=True)))#BiLSTM
    myModel.add(tf.keras.layers.Dropout(0.3))#防止过拟合
    myModel.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(tagSum)))#
    crf = CRF(tagSum, name='crf_layer')#条件随机场
    myModel.add(crf)
    myModel.compile('adam', loss={'crf_layer': crf.get_loss})#优化器adam,条件随机场损失函数


    batch = 150#批处理大小
    import math
    myModel.fit_generator(generate_imp(batch), epochs=20,
                          steps_per_epoch=math.floor(len(train_x) / batch))
    myModel.save("./lstm_1.h5")