# 导⼊必备⼯具包
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# 设置显示⻛格
plt.style.use('fivethirtyeight')
# 分别读取训练tsv和验证tsv
train_data = pd.read_csv("data/train.txt", sep="\t")
valid_data = pd.read_csv("data/dev.txt", sep="\t")
# 获得训练数据标签数量分布
sns.countplot("label", data=train_data)
plt.title("train_data")
plt.show()
# 获取验证数据标签数量分布
sns.countplot("label", data=valid_data)
plt.title("valid_data")
plt.show()

# 在训练数据中添加新的句⼦⻓度列, 每个元素的值都是对应的句⼦列的⻓度
train_data["sentence_length"] = list(map(lambda x: len(x),
                                         train_data["sentence"]))
# 绘制句⼦⻓度列的数量分布图
sns.countplot("sentence_length", data=train_data)
# 主要关注count⻓度分布的纵坐标, 不需要绘制横坐标, 横坐标范围通过dist图进⾏查看
plt.xticks([])
plt.show()
# 绘制dist⻓度分布图
sns.distplot(train_data["sentence_length"])
# 主要关注dist⻓度分布横坐标, 不需要绘制纵坐标
plt.yticks([])
plt.show()
# 在验证数据中添加新的句⼦⻓度列, 每个元素的值都是对应的句⼦列的⻓度
valid_data["sentence_length"] = list(map(lambda x: len(x),
                                         valid_data["sentence"]))
# 绘制句⼦⻓度列的数量分布图
sns.countplot("sentence_length", data=valid_data)
# 主要关注count⻓度分布的纵坐标, 不需要绘制横坐标, 横坐标范围通过dist图进⾏查看
plt.xticks([])
plt.show()
# 绘制dist⻓度分布图
sns.distplot(valid_data["sentence_length"])
# 主要关注dist⻓度分布横坐标, 不需要绘制纵坐标
plt.yticks([])
plt.show()
