#  -*- coding: utf-8 -*-

'''
date:2020-7-29
Author:chensheng
描述：使用LDA主题（聚类算法）实现漏洞预警、事件预警、数据泄露、事件报告分类
'''

import os
import jieba
import jieba.posseg as pseg
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

def filelist(path):
    dir=str(os.getcwd())
    list=os.listdir(path)
    list1=[]
    for i in list:
        list1.append(path+"\\"+i)
    return list1,list


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
    return X_vec

def label():
    y_label=["漏洞预警","事件预警","数据泄露","事件报告"]
    return y_label

def LDA(vec):
    lda = LatentDirichletAllocation(n_components=4,
                                    random_state=56)
    docres = lda.fit_transform(vec)
    lab = label()
    file = filelist("data\\text")[1]
    for i in range(len(file)):
        maxval=max(docres[i])
        list=docres[i].tolist()
        index=list.index(max(list))
        print(str(file[i]) + "文档:" + str(docres[i])+"   类别："+str(lab[index])+"  概率："+str(maxval))

if __name__ == '__main__':
    file=filelist("data\\text")[0]
    print(file)
    docs=readfile(file)
    file1="data\\top_word.txt"
    top=topdoc(file1)
    word=theme(docs,top)
    X_vec=feature_process_word(word)
    LDA(X_vec)