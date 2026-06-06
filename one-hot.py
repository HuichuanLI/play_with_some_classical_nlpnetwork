import joblib
from keras.preprocessing.text import Tokenizer

vocab = {"周杰伦", "陈奕迅", "王力宏", "李宗盛", "吴亦凡", "鹿晗"}
tok = Tokenizer()
tok.fit_on_texts(vocab)
# 生成onehot
for word in vocab:
    idx = tok.texts_to_sequences([word])[0][0]-1
    onehot = [0]*len(vocab)
    onehot[idx] = 1
# 保存加载
# joblib.dump(tok, "./Tokenizer")
# tok = joblib.load("./Tokenizer")