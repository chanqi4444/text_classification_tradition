#coding:utf-8
"""
关键词抽取tf-idf法
要求：python3，NLTK，PyHanLP，fastText
输入：默认一篇文本占一行
"""
from time import time
from fastText import load_model
import numpy as np
from pyhanlp import *
import sys
import codecs
from py.类目关键词抽取_tfidf import extract_keyword

# 实词分词器实例
Term = None
NotionalTokenizer = None
# fastText模型
fasttext_model = None
# 存储所有文本向量的矩阵
text_vec = None
# 存储每篇文本的主题序号
topic_serial = None
# 当前拥有的主题数量
topic_cnt = None
# 每个主题中的文本数量
topic_cnt_dict = None
# 预处理后的文本列表
preprocessed_data = None


# 系统初始化
def init():
    global fasttext_model
    global text_vec
    global topic_serial
    global topic_cnt
    global topic_cnt_dict
    global Term
    global NotionalTokenizer
    global preprocessed_data

    # 读取fastText词语向量矩阵
    fasttext_model = read_fasttext_data('../dictionary/cc.zh.300.bin')
    # 初始化文本向量矩阵
    text_vec = np.array([])
    text_vec.resize((0, fasttext_model.get_dimension()))
    # 初始化文本话题编号序列
    topic_serial = []
    # 初始化话题数量
    topic_cnt = 0
    # 初始化每个主题中的文本数量变量
    topic_cnt_dict = dict()
    # 加载实词分词器 参考https://github.com/hankcs/pyhanlp/blob/master/tests/demos/demo_notional_tokenizer.py
    Term = JClass("com.hankcs.hanlp.seg.common.Term")
    NotionalTokenizer = JClass("com.hankcs.hanlp.tokenizer.NotionalTokenizer")
    #  初始化文本列表
    preprocessed_data = []


# 通用预处理（训练语料和预测语料通用）
def preprocess(text):
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
        # 去掉单字词（这样的词的出现有可能是因为分词系统未登录词导致的）
        if len(word) == 1:
            continue
        word_li.append(word)

    return word_li


# 读取fastText词语向量矩阵
def read_fasttext_data(file_path):
    t0 = time()
    fasttext_model = load_model(file_path)
    t1 = time()
    print("加载fastText向量库时间%.2fs" % (t1-t0))
    return fasttext_model


# 计算句子的单位向量
def compute_sentence_vector(word_li):
    global fasttext_model

    # 初始化句子向量
    sen_vec = np.array([])
    sen_vec.resize((1, fasttext_model.get_dimension()))
    # 在fastText中登陆的词语列表
    has_vec_word_li = []
    for word in word_li:
        # 词语有向量值
        if fasttext_model.get_word_id(word) != -1:
            has_vec_word_li.append(word)
            word_vec = fasttext_model.get_word_vector(word)
            sen_vec += word_vec
    if len(has_vec_word_li) != 0:
        sen_vec /= len(has_vec_word_li)
        # 单位化句子向量
        sen_vec /= np.linalg.norm(sen_vec)
    return sen_vec, has_vec_word_li


# SinglePass文本聚类
def single_pass(sen_vec, sim_threshold=0.6, max_text_number=100):
    global text_vec
    global topic_serial
    global topic_cnt
    if topic_cnt == 0:  # 第1次送入的文本
        # 添加文本向量
        text_vec = np.vstack([text_vec, sen_vec])
        # 话题数量+1
        topic_cnt += 1
        # 分配话题编号，话题编号从1开始
        topic_serial.append(topic_cnt)
        # 初始化话题内文本数量
        topic_cnt_dict[topic_cnt] = 1
    else:  # 第2次及之后送入的文本
        # 文本逐一与已有的话题中的各文本进行相似度计算
        sim_vec = np.dot(sen_vec, text_vec.T)
        # 获取最大相似度值
        max_value = np.max(sim_vec)
        # 获取最大相似度值的文本所对应的话题编号
        topic_ser = topic_serial[np.argmax(sim_vec)]
        print("最相似文本的话题编号", topic_ser, "相似度值", max_value)
        # 添加文本向量
        text_vec = np.vstack([text_vec, sen_vec])
        # 分配话题编号(相似度值大于等于sim_threshold，且话题内文本数量小于等于max_text_number）
        if max_value >= sim_threshold and topic_cnt_dict[topic_ser] <= max_text_number:
            # 将文本聚合到该最大相似度的话题中
            topic_serial.append(topic_ser)
            # 话题内文本数量+1
            topic_cnt_dict[topic_ser] += 1
        else:  # 否则新建话题，将文本聚合到该话题中
            # 话题数量+1
            topic_cnt += 1
            # 将新增的话题编号（也就是增加话题后的话题数量）分配给当前文本
            topic_serial.append(topic_cnt)
            # 初始化话题内文本数量
            topic_cnt_dict[topic_cnt] = 1


def main():
    global preprocessed_data
    global topic_serial

    # 输入文件名
    file_name = sys.argv[1]
    # 资源初始化
    init()
    # 读文本并进行增量聚类
    with codecs.open(file_name, 'rb', 'utf-8') as infile:
        for line in infile:
            line = line.strip()
            if line:
                word_li = preprocess(line, 1)
                sen_vec, has_vec_word_li = compute_sentence_vector(word_li)
                if has_vec_word_li:
                    preprocessed_data.append(u' '.join(word_li))
                    single_pass(sen_vec)
    # 输出聚类结果
    cluster_text_li = []
    outfile_name = file_name.split(u'/')[-1]
    outfile_name = u'Cluster_%s' % outfile_name
    with open(outfile_name, 'wb') as outfile:
        sorted_topic_cnt_li = sorted(topic_cnt_dict.items(), key=lambda x:x[1], reverse=True)
        for out_topic_ser, text_cnt in sorted_topic_cnt_li:
            cluster_text = u''
            if text_cnt >= 5:
                for topic_ser, text in zip(topic_serial, preprocessed_data):
                    if topic_ser == out_topic_ser:
                        out_str = u'%d\t%s\n' % (topic_ser, text)
                        outfile.write(out_str.encode('utf-8', 'ignore'))
                        cluster_text += u'%s ' % text
            if cluster_text:
                cluster_text_li.append(cluster_text)
    # 对每个簇抽取关键词
    category_keywords_li = extract_keyword(cluster_text_li)
    for key_word_li in category_keywords_li:
        print(u','.join(key_word_li))


if __name__ == "__main__":
    main()
