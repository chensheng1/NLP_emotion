#  -*- coding: utf-8 -*-

'''
date:2020-7-30
Author:chensheng
描述：使用协同过滤，根据文档的词频，时间，厂商，级别，影响，
'''
import os
import jieba
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


#先分词，提取出主题词、时间、级别、厂商，然后在计算内容相似度，用户相似度，将基于内容和用户取平均权值，选择前10推荐，然后筛选去重，推荐

#读取数据摘要
def readfile(file):
    directory = str(os.getcwd())
    file1 = []
    docs = []
    for filename in file:
        filepath = os.path.join(directory, filename)
        file1.append(filepath)
    for f in file1:
        docs.append(open(f, encoding="utf-8", errors="ignore").read())
    return docs

#读取停用词、厂商
def topdoc(file):
    directory = str(os.getcwd())
    filepath = os.path.join(directory, file)
    result = []
    fr = open(filepath, encoding='utf-8')
    for line in fr.readlines():
            line = line.rstrip("\n")
            result.append(line)
    return result

#读取厂商信息
def title(file,store):
    list = []
    result1=[]
    for doc in file:
        result = jieba.cut(doc)
        word = []
        for i in result:
            if len(i) > 1 :
                word.append(i)
        result1.append(word)
    for x in result1:
        store1 = set()
        for le in range(0,len(x)-1):
            if x[le] in store:
                store1.add(x[le])
        if len(store1)==0:
            cha="其它"
            store1.add(cha)
        list.append(store1)
    return list

#读取时间,得到时间维度



#得到去除停用词的文本信息
def participle(docs,top):
    result1=[]
    list=[]
    for doc in docs:
        result=jieba.cut(doc)
        word = []
        for i in result:
            if len(i)>1 and i not in top:
                word.append(i)
        result1.append(word)
    for y in result1:
        str=' '.join(y)
        list.append(str)
    return list

#词频向量化
def feature_process_word(data):
    vectorizer = TfidfVectorizer(min_df=0.0, analyzer="word",lowercase=False, sublinear_tf=True,
                                 ngram_range=(1, 3))
    X_vec = vectorizer.fit_transform(data)
    return X_vec

#计算余玄相似度
def bit_product_sum(x, y):
    return sum([item[0] * item[1] for item in zip(x, y)])
def cosine_similarity(x, y, norm=False):

    cos = bit_product_sum(x, y) / (np.sqrt(bit_product_sum(x, x)) * np.sqrt(bit_product_sum(y, y)))
    return cos

def simil(data):
    length=len(data)
    for i in range(length):
        for j in range(i+1,length):
            x=data[i]
            y=data[i+1]
            probability=cosine_similarity(x,y)
            print("第"+str(i+1)+"篇文档与第"+str(j+1)+"文档的相似度："+str(probability))


if __name__ == '__main__':
    file=['data\\test.txt', 'data\\test2.txt']
    docs=readfile(file)
    file1="data\\top_word.txt"
    top = topdoc(file1)
    word=participle(docs,top)
    X_vec = feature_process_word(word)
    simil(X_vec.toarray())


