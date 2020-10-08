from nltk import data
import os
import nltk
path="./txts"#txt文件路径
outputpath="./datas"#数据集输出路径


"""
根据txt文件和ann文件生成项目需要的数据集
"""


def getOneFileData(file):
    txtpath=os.path.join(path,file)#获得全路径
    annpath=os.path.join("./anns",file[0:file.rfind('.')]+".ann")#或得对应的ann文件路径
    txt_now=readOneFile(txtpath)#读取文件内容
    ann_now=readOneFile(annpath)
    entity = dealanns(ann_now)#解析一个ann文件获取全部实体
    sens = dealtxts(txt_now)#解析一个txt文件获得全部句子
    res = []
    for now in sens:
        tmp = changeOneSen(now, entity)#处理一个句子
        res.append(tmp)
    filename = os.path.join(outputpath, file)
    writeFile("\n".join(res), filename)




def changeOneSen(sen,entity):

    def getOneentity(start,end,entity):#查找start到end对应的单词是否属于某个实体，entity:当前文章的实体表
        for now in entity:
            if int(start)>=int(now[1]) and int(end)<=int(now[2]):
                return now[0]
        return "others"


    sen_index=sen[1]#获取句子在文章中的开始索引
    sen[0]=sen[0].replace("\n"," ")#去掉换行
    tmp_words = nltk.word_tokenize(sen[0])#分词
    words=[]
    now_index = 0
    """下面一个循环将一个句子中的所有单词的起始位置找到"""
    for now_word in tmp_words:
        imp = sen[0].find(now_word, now_index)
        now_index = imp + len(now_word)
        words.append([now_word, sen_index+imp])
    res_entity=[]#保存一个句子对应的各个单词的实体
    for word in words:
        start_index=word[1]
        final_index=start_index+len(word[0])
        tmp_en=getOneentity(start_index,final_index,entity)
        res_entity.append(tmp_en)
    str_ens=" ".join(res_entity)
    return sen[0]+"\t\t"+str_ens#使用两个tab分割



def dealanns(ann):#解析一个ann文件
    line_list=ann.split("\n")
    res=[]
    for line in line_list:
        if len(line)>0 and line[0]=="T":
            tmp=line.split("\t")
            tmp2=tmp[1].split(" ")
            res.append(tmp2)
    return res

def dealtxts(txt):
    tmp = nltk.tokenize.sent_tokenize(txt)
    sens_index = 0
    res=[]
    for now_sen in tmp:
        start=txt.find(now_sen,sens_index)
        sens_index=start+len(now_sen)
        res.append([now_sen,start])
    return res

def readOneFile(arg_path):
    f=open(arg_path,"r+",encoding='utf-8')
    content=f.read()
    f.close()
    return content

def writeFile(content,out):
    f=open(out,"w+",encoding='utf-8')
    f.write(content)
    f.close()

if __name__=="__main__":
    data.path.append("./nltk_data")#nltk文件
    files = os.listdir(path)#扫描路径下的全部文件
    for file in files:
        print(file)
        getOneFileData(file)#处理一个文件