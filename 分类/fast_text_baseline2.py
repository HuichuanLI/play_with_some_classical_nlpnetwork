import fasttext
import time
import os
import sys
import jieba

id_to_label = {}
idx = 0
with open('../data/class.txt', 'r', encoding='utf-8') as f1:
    for line in f1.readlines():
        line = line.strip('\n').strip()
        id_to_label[idx] = line
        idx += 1
print('id_to_label:', id_to_label)
count = 0
train_data = []

with open('../data/test.txt', 'r', encoding='utf-8') as f2:
    for line in f2.readlines():
        line = line.strip('\n').strip()
        sentence, label = line.split('\t')
        # 1: ⾸先处理标签部分
        label_id = int(label)
        label_name = id_to_label[label_id]
        new_label = '__label__' + label_name
        # 2: 然后处理⽂本部分, 为了便于后续增加n-gram特性, 可以按字划分, 也可以按词划分
        sent_char = ' '.join(jieba.lcut(sentence))
        # 3: 将⽂本和标签组合成fasttext规定的格式
        new_sentence = new_label + ' ' + sent_char
        train_data.append(new_sentence)
        count += 1
        if count % 10000 == 0:
            print('count=', count)

with open('../data/train_fast.txt', 'w', encoding='utf-8') as f3:
    for data in train_data:
        f3.write(data + '\n')
print('FastText训练数据预处理完毕!')

train_data_path = '../data/train_fast.txt'
dev_data_path = './data/dev_fast.txt'
test_data_path = './data/test_fast.txt'

# 开启模型训练
# autotuneValidationFile参数需要指定验证数据集所在的路径
# 它将在验证集是使用随机搜索的方法寻找最优的超参数
# 使用autotuneDuration参数可以控制随机搜索的时间, 默认是300秒.
# 根据不同的需求, 可以延长或者缩短时间.
# verbose: 该参数决定日志打印级别, 当设置为3, 可以将当前正在尝试的超参数打印出来
model = fasttext.train_supervised(input=train_data_path,
                                  autotuneValidationFile=train_data_path,
                                  autotuneDuration=600,
                                  wordNgrams=2,
                                  verbose=3)

# 开启模型测试
result = model.test(test_data_path)
print(result)

# 模型保存
time1 = int(time.time())
model_save_path = "./toutiao_fasttext_{}.bin".format(time1)
model.save_model(model_save_path)
