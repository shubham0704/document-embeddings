from __future__ import print_function
import numpy as np
import sys
import os
sys.path.append('/home/master/models/syntaxnet/')
import svo_extractor
#all_sents = np.load('useful_sents.npy')
#import gensim.models.word2vec as w2v
#model = w2v.Word2Vec.load(os.path.join("trained","corpus0vec.w2v"))

svo_sents = []
'''
for i, sent in enumerate(all_sents[30000:]):
	svo_sents.append(svo_extractor.get_svo(sent))
'''	
'''
for i,sent in enumerate(all_sents[6795:6796]):
    # check if you would like to use this
    #if len(sent)>40: its btw 6795 to 6796 or sentence 6795 this sentence is 0
    #   continue
    if sent=='\n':
        print 'gotcha'
        print 'the sentence is --> ', sent,'its length is ->', len(sent)
        #svo_sents.append(svo_extractor.get_svo(sent))
    else:
        useless_sents.append(i)
'''
s1 = 'you cannot believe in god until you believe in yourself'
s1_svo = svo_extractor.get_svo(s1)

import networkx as nx


def gen_graph(sentence):
	
	graph = nx.Graph()
	for i, token in enumerate(sentence.token):
		
		node_id = i
		graph.add_node(node_id, label=token.word)
		
		if token.head >= 0:
		  src_id = token.head
		  graph.add_edge(
			  src_id,
			  node_id,
			  label=token.label,
			  key="parse_{}_{}".format(node_id, src_id))
	return graph

print (s1_svo[3])
graph = gen_graph(s1_svo[3])
np.save('node2vec/src/nodes.npy', graph.nodes(data=True))
	
nx.write_edgelist(graph, "node2vec/graph/sentence_test.edgelist", data=True)
