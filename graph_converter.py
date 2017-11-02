import numpy as np
import sys, os
sys.path.append(os.path.expanduser('~') + '/models/syntaxnet/')
import tree_gen
import networkx as nx
import matplotlib.pyplot as plt
from node2vec.src.model_maker import model_maker

# TODO make a script to place svo_extractor into the syntaxnet directory
class Args:
	def __init__(self, graph):
		self.graph = graph
		self.input = '/home/master/Desktop/GIT/node2vec-final/node2vec/graph/karate.edgelist' 
		self.output = '/home/master/Desktop/GIT/node2vec-final/node2vec/emb/karate.emb'
		self.dimensions = 128
		self.walk_length = 80
		self.num_walks = 10
		self.window_size = 10
		self.iter = 1
		self.workers = 8
		self.p = 1
		self.q = 1
		self.weighted = 0
		self.directed = 0 
		
sents = np.load("useful_sents.npy")

tokenised_sents = [[word for word in sent.split()] for sent in sents]

vocab = {}
for sent in tokenised_sents:
    for word in sent:
        if word not in vocab:
            vocab[word] = 0
        else:
            vocab[word] += 1

unique_words = vocab.keys()
unique_dict = {}

for i, word in enumerate(unique_words):
    unique_dict[word] = i

new_sents = []
for sent in tokenised_sents:
    new_sent = []
    for word in sent:
        new_sent.append(unique_dict[word])
    new_sents.append(new_sent)


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

    # let the graph build as default just change node-ids after graph formation
    glob_graph = nx.Graph()
    # traverse graph add nodes and edges
    for node in graph.nodes():
		try:
			node_id = unique_dict[graph.node[node]['label']]
			glob_graph.add_node(node_id, label=graph.node[node]['label'])
		except:
			pass
			
    for edge in graph.edges():
		try:
			src_id = unique_dict[graph.node[edge[0]]['label']]        
			node_id = unique_dict[graph.node[edge[1]]['label']]
			glob_graph.add_edge(
				  src_id,
				  node_id,
				  key="parse_{}_{}".format(node_id, src_id))
		except:
			pass
    return glob_graph
'''
# sanity check for one sentence
s = 'you cannot believe in god until you believe in yourself'
result = svo_extractor.get_svo(s)
parse_tree = result[3]
print (parse_tree)
graph = gen_graph(parse_tree)


# checking sanity by plotting
pos = nx.spring_layout(graph)
label_dict = {}
for node in graph.nodes():
	label_dict[node] = unique_words[node]

nx.draw(graph, labels=label_dict, with_labels=True)
plt.show()
# checking done
'''
def plot_graph(graph):
	pos = nx.spring_layout(graph)
	label_dict = {}
	for node in graph.nodes():
		label_dict[node] = unique_words[node]

	nx.draw(graph, labels=label_dict, with_labels=True)
	plt.show()

limit = 150
graphs = []
for sent in sents[:limit]:
	
	parse_tree = tree_gen.get_tree(sent)
	graph = gen_graph(parse_tree)
	# now time to concatenate all the graphs
	graphs.append(graph)

giant_graph = graphs[0]
for i in range(1, len(graphs)):
	giant_graph = nx.compose(graphs[i], giant_graph)

#plot_graph(giant_graph)


args = Args(giant_graph)
model = model_maker(args)

sent_embeddings = []
for sent in new_sents[:limit]:
	emb = np.zeros((args.dimensions, 1))
	for word in sent:
		try:
			emb += model[str(word)].reshape((-1, 1))			
		except:
			pass
			#print str(word), unique_words[word], 'not found in model vocab'	
	sent_embeddings.append(emb)

np.save('embs-150.npy', sent_embeddings)

