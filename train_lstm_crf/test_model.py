import keras
from CRF import CRF
import nltk
import numpy as np
import gensim
import nltk

"""
lstm的测试，将字符串input的内容标注
"""

maxSenLen=100
WORD_WIDTH=100
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
input='Google.'
words=nltk.word_tokenize(input)#分词
model = gensim.models.Word2Vec.load("word2vec_model")#加载word2vec
"""下面两行是word2vec的增量训练"""
model.build_vocab([words], update=True)
model.train([words], total_examples=model.corpus_count, epochs=model.iter)
"""下面三行将单词转化为词向量"""
n=np.zeros((1,maxSenLen,WORD_WIDTH))
for i in range(len(words)):
    n[0][i]=model[words[i]]

#加载lstm
model=keras.models.load_model("./lstm_5.h5",custom_objects={'CRF': CRF,"get_loss":CRF.get_loss})
#预测
res=model.predict(n)
for i in range(len(words)):
    print(words[i]+" "+entities[res[0][i]])