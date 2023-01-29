# 垃圾邮件分类系统
# 22210980051 李韵

from os import walk
from random import shuffle
from collections import Counter
from sklearn import preprocessing
import pandas as pd
import nltk

def get_data(path):
    # 读取所有邮件数据
    pathwalk = walk(path)

    allHamData = []
    allSpamData = []
    for root, dr, file in pathwalk:
        if 'ham' in str(file):
            for obj in file:
                with open(root + '/' + obj, encoding='latin1') as ip:
                    allHamData.append(" ".join(ip.readlines()))
                    
        elif 'spam' in str(file):
            for obj in file:
                with open(root + '/' + obj, encoding='latin1') as ip:
                    allSpamData.append(" ".join(ip.readlines()))
    # 去重
    allHamData = list(set(allHamData))
    allSpamData = list(set(allSpamData))
    return allHamData, allSpamData

def preprocess(data):
    # 分词
    tokens = nltk.word_tokenize(data)
    tokens = [w.lower() for w in tokens if w.isalpha()]
    # 寻找不常见的词
    cnt = Counter(tokens)
    uncommons = cnt.most_common()[:-int(len(cnt)*0.1):-1]
    # 获取停用词
    stops = set(nltk.corpus.stopwords.words('english'))
    # 去除不常见的词和停用词
    tokens = [w for w in tokens if (w not in stops and w not in uncommons)]
    # 词形还原
    lemmatizer = nltk.WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(w, pos='a') for w in tokens]

    return tokens

if __name__ == '__main__':

    # 读取所有邮件数据
    path = r"./enron-spam/"
    allHamData, allSpamData = get_data(path)

    # 数据处理
    allData = allHamData + allSpamData
    labels = ['ham'] * len(allHamData) + ['spam'] * len(allSpamData)
    raw_df = pd.DataFrame({'email': allData, 'label': labels})

    nltk_processed_df = pd.DataFrame()
    nltk_processed_df['email'] = [preprocess(e) for e in raw_df['email']]

    label_encoder = preprocessing.LabelEncoder()
    nltk_processed_df['label'] = label_encoder.fit_transform(raw_df['label'])

    # 训练/测试数据划分
    X, y = nltk_processed_df['email'], nltk_processed_df['label']
    X_featurized = [Counter(i) for i in X]
    allDataProcessed = [(X_featurized[i], y[i]) for i in range(len(X))]
    shuffle(allDataProcessed)
    trainData, testData = allDataProcessed[:int(len(allDataProcessed)*0.7)], allDataProcessed[int(len(allDataProcessed)*0.7):]

    # 模型训练
    model = nltk.classify.NaiveBayesClassifier.train(trainData)
    accuracy = nltk.classify.accuracy(model, testData)
    print("基于朴素贝叶斯模型的垃圾邮件分类器的准确率为: ", accuracy)

    # 输入邮件内容进行预测
    input_email = input("请输入邮件内容: ")
    print("输入的邮件内容为: "+input_email)

    # 输出分类结果
    result = model.classify(Counter(preprocess(input_email)))
    result = list(label_encoder.inverse_transform([result]))[0]
    print("鉴定结果为: "+result+(' 垃圾邮件' if result=='spam' else ' 正常邮件'))
