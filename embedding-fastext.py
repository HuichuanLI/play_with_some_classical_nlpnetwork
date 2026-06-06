import fasttext

model1 = fasttext.train_unsupervised('data/fil9')
model = fasttext.load_model("data/fil9.bin")

model.get_nearest_neighbors('sports')

model.get_nearest_neighbors('music')

model.get_nearest_neighbors('dog')
