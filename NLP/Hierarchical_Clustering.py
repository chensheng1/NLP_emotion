#  -*- coding: utf-8 -*-

'''
date:2020-8-4
Author:chensheng
描述：使用k-means（聚类算法）实现漏洞预警、事件预警、数据泄露、事件报告分类
'''

import os
import jieba
import jieba.posseg as pseg
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering

def readfile(file):
    directory = str(os.getcwd())
    file1=[]
    docs=[]
    for filename in file:
        filepath = os.path.join(directory, filename)
        file1.append(filepath)
    for f in file1:
        docs.append(open(f, encoding="utf-8", errors="ignore").read())
    return docs

def topdoc(file):
    directory = str(os.getcwd())
    filepath = os.path.join(directory, file)
    result = []
    fr = open(filepath, encoding='utf-8')
    for line in fr.readlines():
            line = line.rstrip("\n")
            result.append(line)
    return result

def theme(docs,top):
    result1=[]
    list=[]
    for doc in docs:
        result=pseg.cut(doc)
        word = []
        for i in result:
            print(i.word,i.flag)
            word_s=['n','eng']
            if len(i.word)>1 and i.word not in top and i.flag in word_s:
                word.append(i.word)
        result1.append(word)
    for y in result1:
        str=' '.join(y)
        list.append(str)
    return list

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

def feature_process_word(data):
    vectorizer = TfidfVectorizer(min_df=0.0, analyzer="word",lowercase=False, sublinear_tf=True,
                                 ngram_range=(1, 3))
    X_vec = vectorizer.fit_transform(data)
    dic=vectorizer.vocabulary_
    L = sorted(dic.items(), key=lambda item: item[1], reverse=True)
    print(L[:10])
    return X_vec

def label():
    y_label=["漏洞预警","事件预警","数据泄露","事件报告"]
    return y_label

def k_means(vec):
    linkages = ['ward', 'average', 'complete']
    n_clusters_ = 3
    lda = AgglomerativeClustering(linkage=linkages[2],n_clusters = n_clusters_)
    docres = lda.fit_predict(vec.toarray())
    print(docres)

if __name__ == '__main__':
    file=['data\\test.txt', 'data\\test2.txt','data\\test3.txt']
    docs=readfile(file)
    file1="data\\top_word.txt"
    top=topdoc(file1)
    word=theme(docs,top)
    X_vec=feature_process_word(word)
    k_means(X_vec)