"""
The purpose of this file is to run 2 models on the same data

1. Our meta-path algorithm
2. Word2Vec algorithm

on 8 iterations of data adding 100 mb of training data at each iteration
save the score on the analogy type after each iteration into a list.
Plot the x and y at the end of the iterations

"""
import multiprocessing
from time import time

from graph_builder import GraphBuilder, Args
from sentence_loader import lazy_load
from gensim.models.word2vec import w2v

def metaN2V_model(argList=[None]):
    args = Args(argList)
    G = GraphBuilder()
    return G

# hyperparameters for Word2Vec
num_features = 150
min_word_count = 1
num_workers = multiprocessing.cpu_count()
context_size = 6
down_sampling = 1e-3
seed = 1

w2v_model = w2v.Word2Vec(
    sg=1,
    seed=seed,
    workers=num_workers,
    size=num_features,
    min_count=min_word_count,
    window=context_size,
    sample=downsampling
)

G = metaN2V_model()

def test_w2v(w2v_model, tokenized_sents):
        w2v_model.build_vocab(tokenized_sents)
        w2v_model.train(tokenized_sents,total_examples=w2v_model.corpus_count,epochs=w2v_model.iter)

tokenized_sents, sents = lazy_load()
prev = time()
test_w2v(w2v_model, tokenized_sents)
now = time() - prev
print("It took {} time".format(now))
#now run both models for k iteration
# for i in range(k):
#     tokenized_sents, sents = lazy_load()
#     # setup metapath2vec algorithm
#     G.gen_data(sents, tokenized_sents)
#     G.gen_giant_graph()
#     args.graph = G.giant_graph
# 	args.meta_paths = G.meta_paths
# 	model = model_maker(args, G.unique_words) # here we have to again and again train , think of a way of checkpointing
#     # setup word2vec algorithm
#     model.save('metaN2V_model_'+ str(i+1)+ '00MB')
#     w2v_model.build_vocab(tokenized_sents)
#     w2v_model.train(tokenized_sents,total_examples=w2v_model.corpus_count,epochs=w2v_model.iter)
