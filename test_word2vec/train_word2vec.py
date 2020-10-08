import nltk
import os
import gensim
'''
word2vec的训练，使用txt文件创建预料库
'''


nltk.data.path.append('./nltk_data')#加载nltk
path="./txts"#txt文件路径
words=[]#保存单词的list


def creat_words():
    global words
    files = os.listdir(path)
    for file in files:
        txt_path=os.path.join(path,file)
        f=open(txt_path,"r+")
        content=f.read()
        f.close()
        sens=nltk.sent_tokenize(content)#分句
        for now in sens:
            words.append(nltk.tokenize.word_tokenize(now))#分词后添加到word里面
    print(len(words))

if __name__=='__main__':
    creat_words()
    '''
    word2vec的训练，生成大小为100的词向量，窗口大小5，最小词频1，工作线程5
    '''
    model = gensim.models.Word2Vec(words, size=100, window=5,
                           min_count=1, workers=5)
    model.save("word2vec_model")#保存模型文件