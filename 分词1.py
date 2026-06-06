import jieba
import jieba.posseg as pseg

# 三种分词模式
text = "无线电法国别研究"
cut_exact = jieba.lcut(text, cut_all=False)  # 精确
cut_all = jieba.lcut(text, cut_all=True)  # 全匹配
cut_search = jieba.lcut_for_search(text)  # 搜索引擎

# 自定义词典：userdict.txt内容：八一双⿅ 3 nz
# jieba.load_userdict("userdict.txt")
# custom_cut = jieba.lcut("八一双⿅更名为八一南昌篮球队!")

# 词性标注
pos_res = pseg.lcut("我爱北京天安门")
