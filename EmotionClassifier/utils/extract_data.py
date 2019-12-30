"""
*.xml -> *.txt  format = sentence + '\t' + emotion_type + '\n'

@author: Zhenwei Wang
@date: 12/29/2019
"""
# coding=utf-8

# 通过minidom解析xml文件
import xml.dom.minidom as xmldom
import os

# 待读取的xml文件
xmlfilepath_1 = os.path.join('..', 'data', 'nlpcc2014', 'Training data for Emotion Classification.xml')
xmlfilepath_2 = os.path.join('..', 'data', 'nlpcc2014', 'EmotionClassficationTest.xml')
xmlfilepath_3 = os.path.join('..', 'data', 'nlpcc2013', '微博情绪样例数据V5-13.xml')
xmlfilepath_4 = os.path.join('..', 'data', 'nlpcc2013', '微博情绪标注语料.xml')
xmlfilepath_5 = os.path.join('..', 'data', 'nlpcc2014', 'NLPCC2014微博情绪分析样例数据.xml')

data = {}
emotion_num = {'anger': 0, 'disgust': 0, 'fear': 0, 'happiness': 0, 'like': 0, 'sadness': 0, 'surprise': 0,
               'other': 0}
# emotion_num = {'愤怒': 0, '厌恶': 0, '恐惧': 0, '高兴': 0, '喜好': 0, '悲伤': 0, '惊讶': 0,
#                'other': 0}
for xmlfilepath in [xmlfilepath_5]:
    print('Start parsing ', xmlfilepath)
    # 得到文档对象
    domobj = xmldom.parse(xmlfilepath)
    # 得到元素对象
    elementobj = domobj.documentElement
    # 获得子标签
    subElementObj = elementobj.getElementsByTagName("sentence")

    print('There are ' + str(len(subElementObj)) + ' sentences in current xml file')
    # 获得标签属性值
    for sentence in subElementObj:
        emotion_tag = sentence.getAttribute('opinionated')
        if not emotion_tag:
            emotion_tag = sentence.getAttribute('emotion_tag')

        assert emotion_tag in ['Y', 'N']

        if emotion_tag == 'N':
            label = 'other'
        elif emotion_tag == 'Y':
            label = sentence.getAttribute('emotion-1-type')

            assert label in emotion_num.keys()

        text = sentence.firstChild.data
        if text in list(data.keys()) or len(text) < 5:
            pass
        else:
            data[text] = label
            emotion_num[label] = emotion_num[label] + 1
    print('There are ' + str(len(data)) + ' sentences in total')

print('saving.....')
out = open(os.path.join('..', 'data', 'raw_data.txt'), 'w')
for text, label in data.items():
    out.write(text + '\t' + label + '\n')
print('success saving to data/raw_data.txt')

print('sentences: ', len(data))
for label, num in emotion_num.items():
    print(label, ': ', num, ' ', num / len(data) * 100, '%')
