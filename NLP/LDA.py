#  -*- coding: utf-8 -*-
import jieba

filepaths = ['C:\\Users\\asus\\Desktop\\NLP\\data\\test.txt', 'C:\\Users\\asus\\Desktop\\NLP\\data\\test2.txt']

docs = [open(f,encoding="utf-8",errors="ignore").read() for f in filepaths]  # 文档
print(docs)
docs = [jieba.cut(doc)
        for doc in docs]  # 分词

docs = [[w
         for w in doc
         if len(w) > 1]
        for doc in docs]  # 本文中去停止词操作比较简单,只保留词语长度大于1的。
print(docs)
corpus = [' '.join(doc)
          for doc in docs]  # 处理之后，每个文档的词语列表，加上空格
print("分词：", corpus)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(corpus)
print("词频统计矩阵:", tfidf_matrix)  # 词频统计，4*496
# 4行指的是四个文档
# 496列是496个词语（也就是语料中一共出现了496个词语）


# '''
# 由于LDA属于聚类分析的一种，而聚类分析过程中会随机初始化，为了保证你也能得到与大邓一样的运行结果，我设置了random_state=123456。
# 当然设置成别的数字也可以，这里的random_state相当于口令，咱们两个口令一致才能得到相同的答案。如果你换了random_state，
# 那么咱们两个得到的结果可能会有出入。
# '''
lda = LatentDirichletAllocation(n_components=4,
                                random_state=56)  # n_components，由于我们有预先的知识，知道这四个文档来源于三国和三体，所以话题数K天然的等于2
docres = lda.fit_transform(tfidf_matrix)
print("分类结果矩阵：", docres)  # 4*2，4个文档分别属于两个话题的概率，取其中概率大的