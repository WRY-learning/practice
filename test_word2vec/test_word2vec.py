import gensim
"""
模型的测试，找出和the最相近的10个单词
"""
model = gensim.models.Word2Vec.load("word2vec_model")
res=model.most_similar('the')
print(res)