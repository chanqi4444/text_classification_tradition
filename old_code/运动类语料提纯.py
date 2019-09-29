#coding:utf-8
"""
对运动类语料提纯
"""

import codecs
import os

stay_corpus_set = set([1, 2, 21, 12, 6, 14, 18, 39, 24, 35,
9, 8, 31, 15, 29, 13, 25, 32, 28, 40, 20,
10, 41, 16, 17, 19, 4, 36])

# 清空目录下所有文件
def del_file(path):
    ls = os.listdir(path)
    for i in ls:
        c_path = os.path.join(path, i)
        if os.path.isdir(c_path):
            del_file(c_path)
        else:
            os.remove(c_path)

del_file('data/new_sports')

with codecs.open('data/res_single_pass.txt', 'rb', 'utf-8', 'ignore') as infile:
    text_cnt = 0
    for line in infile:
        line = line.strip()
        if line:
            cluster_ser, text = line.split(u'\t')
            if int(cluster_ser) in stay_corpus_set:
                with open('data/new_sports/%d.txt'%(text_cnt), 'wb') as outfile:
                    out_str = u'%s' % line
                    outfile.write(out_str.encode('gbk', 'ignore'))
                text_cnt += 1
