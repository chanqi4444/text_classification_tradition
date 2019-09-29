#coding:utf-8
"""
关键词抽取tf-idf法
用法：python 类目关键词抽取tf.py 文件名 每个类目最大关键词数量
要求：python3，sklearn，PyHanLP
说明：输入文件中每一行存储一个类目的所有文本。
程序会统计每个词项的tf-idf值，这里的idf指的逆类目频率，
并输出每个类目的按tf-idf值降序的topx个词语，x由第2个参数决定默认为10
"""

import codecs
from pyhanlp import *
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import ngrams

# 加载实词分词器 参考https://github.com/hankcs/pyhanlp/blob/master/tests/demos/demo_notional_tokenizer.py
Term = JClass("com.hankcs.hanlp.seg.common.Term")
NotionalTokenizer = JClass("com.hankcs.hanlp.tokenizer.NotionalTokenizer")


# 通用预处理（训练语料和预测语料通用）
def preprocess(text):
    # 如果抽取词语则n_grams=1，如果抽取2grams则n_grams=2
    n_grams = 2
    # 全部字母转小写
    text =text.lower()
    word_li = []

    #  NotionalTokenizer.segment中有去除停用词的操作
    for term in NotionalTokenizer.segment(text):
        word = str(term.word)
        pos = str(term.nature)
        # 去掉时间词
        if pos == u't':
            continue
        word_li.append(word)

        # 去掉单字词（这样的词的出现有可能是因为分词系统未登录词导致的）
        if n_grams == 1 and len(word) == 1:
            continue
        word_li.append(word)

    # 如果只是分词则直接返回word_li即可
    if n_grams == 2:
        ngrams2_li = [u'_'.join(w) for w in ngrams(word_li, 2)]
        word_li = ngrams2_li
    return word_li


def extract_keyword(text_li, topx=10):
    """
    用tf-idf法抽取每个类目的关键词
    :param text_li: 类目文本类表，每个元素表示一个类目的所有文本串
    :param topx: 每个类目抽取出的关键词数量
    :return: 返回每个类目的关键词序列
    """
    tv = TfidfVectorizer(analyzer=preprocess)
    tv_fit = tv.fit_transform(text_li)
    vsm = tv_fit.toarray()
    category_keywords_li = []
    for i in range(vsm.shape[0]):
        sorted_keyword = sorted(zip(tv.get_feature_names(), vsm[i]), key=lambda x:x[1], reverse=True)
        category_keywords = [w[0] for w in sorted_keyword[:topx]]
        category_keywords_li.append(category_keywords)
    return category_keywords_li


def main():
    input_file_name = sys.argv[1]
    if len(sys.argv) == 3:
        topx = int(sys.argv[2])
    else:
        topx = 10
    with codecs.open(input_file_name, 'rb', 'utf-8', 'igonre') as infile:
        text_li = infile.readlines()
    category_keywords_li = extract_keyword(text_li, topx)
    print(category_keywords_li)


if __name__ == "__main__":
    main()
