import numpy as np
import sys, os
sys.path.append(os.path.expanduser('~') + '/models/syntaxnet/')
import tree_gen
import networkx as nx
import matplotlib.pyplot as plt
from node2vec.src.model_maker import model_maker
from tensor_visualize import visualize
from sentence_loader import lazy_load
from memory_profiler import profile


class Args:
	
	def __init__(self, graph=None, dimensions=100, walk_length=80, num_walks=10, 
						window_size=10, iter=1, workers=8, p=1, q=1, weighted=0, directed=0, **extra_kwargs):
		self.graph = graph
		#self.input = './node2vec/graph/useful_sents.npy' 
		self.input  = './node2vec/graph/sents.npy'
		self.output = './node2vec/emb/emb.npy'
		self.dimensions = dimensions
		self.walk_length = walk_length
		self.num_walks = num_walks
		self.window_size = window_size
		self.iter = iter
		self.workers = workers
		self.p = p
		self.q = q
		self.weighted = weighted
		self.directed = directed
		self.meta_paths = None

	
	def __str__(self):
		return """printing initialized args list -
					input: {0}
					output: {1}
					dimensions: {2}
					walk_length: {3}
					num_walks: {4}
					window_size: {5} 
					iter: {6}
					workers: {7}
					p: {8}
					q: {9}
					graph: {10}
					meta_paths: {11}
				""".format(self.input,
						 self.output,
						 self.dimensions,
						 self.walk_length,
						 self.num_walks,
						 self.window_size,
						 self.iter,
						 self.workers,
						 self.p,
						 self.q,
						 self.graph,
						 self.meta_paths
						) 
		
class GraphBuilder:
	
	new_sents = []
	unique_dict = {} # stores word, id pair
	graphs = []
	unique_words = {}
	sents = None
	meta_paths = set()
	giant_graph = nx.Graph()
	done = 0
	id_no = 0

	def __init__(self, limit=150):
		self.limit = limit
		
	
	def gen_data(self, sents=None):
		"""
		populates the vocabulary and initializes all class variables
		"""
		if sents:
			tokenised_sents = [[word for word in sent.split()] for sent in sents]
		else:
			tokenised_sents, self.sents = lazy_load()
			if not tokenised_sents:
				self.done = 1
		for sent in tokenised_sents:
			for word in sent:
				word = word.lower()
				if word not in self.unique_dict:
					self.unique_dict[word] = self.id_no
					self.unique_words[self.id_no] = word
					self.id_no += 1

		for sent in tokenised_sents:
			new_sent = []
			for word in sent:
				new_sent.append(self.unique_dict[word])
			self.new_sents.append(new_sent)


	def gen_graph(self, sentence):
		"""
		given a sentence generate a graph with global indices
		"""
		graph = nx.Graph()
		path = []
		for i, token in enumerate(sentence.token):

			node_id = i
			# TODO -  change token.word to token.label
			# graph.add_node(node_id, label=token.word, tag=token.label)
			graph.add_node(node_id, {'label':token.word, 'tag':token.label})
			
			# for each sentence extract the inner grammatical structure
			path.append(token.label)

			if token.head >= 0:
			  src_id = token.head
			  graph.add_edge(
				  src_id,
				  node_id,
				  label=token.label,
				  key="parse_{}_{}".format(node_id, src_id))

		path = tuple(path)
		self.meta_paths.add(path)
		# let the graph build as default just change node-ids after graph formation
		glob_graph = nx.Graph()
		# traverse graph add nodes and edges
		for node in graph.nodes():
			try:
				node_id = self.unique_dict[graph.node[node]['label']]
				#print "I know tags", graph.node[node]['tag'],'\n'
				glob_graph.add_node(node_id, {'label':graph.node[node]['label'], 'tag':graph.node[node]['tag']})
			except:
				pass
				
		for edge in graph.edges():
			try:
				src_id = self.unique_dict[graph.node[edge[0]]['label']]        
				node_id = self.unique_dict[graph.node[edge[1]]['label']]
				glob_graph.add_edge(
					  src_id,
					  node_id,
					  key="parse_{}_{}".format(node_id, src_id))
			except:
				pass
		return glob_graph
		
	@profile	
	def gen_giant_graph(self):
		
		"""
		generate sentence graphs and concatenate them 
		"""
		sents = self.sents
		for sent in self.sents:
			self.giant_graph = nx.compose(self.get_graph(sent), self.giant_graph)

		

	def get_graph(self, sent):

		"""
		takes a sentence as input return a graph
		"""
		parse_tree = tree_gen.get_tree(sent)
		graph = self.gen_graph(parse_tree)
		return graph


	
	def get_embeddings(self, model, dims):
		
		sent_embeddings = []
		for sent in self.new_sents[:self.limit]:
			emb = np.zeros((dims, 1))
			for word in sent:
				try:
					emb += model[str(word)].reshape((-1, 1))			
				except:
					pass
					#print str(word), unique_words[word], 'not found in model vocab'	
			sent_embeddings.append(emb)
		return sent_embeddings
		#np.save(args.output, sent_embeddings)

	def plot_graph(self,graph):
		pos = nx.spring_layout(graph)
		label_dict = {}
		for node in graph.nodes():

			label_dict[node] = self.unique_words[node]

		nx.draw(graph, labels=label_dict, with_labels=True)
		plt.show()
		pass

	def plot_d3graph(self, graph):



		pass
	def gen_graph_one(self, sentence):

	    graph = nx.Graph()
	   #  node_dict = {}
	   #  for i, token in enumerate(sentence.token):

	   #      node_id = i
	   #      node_dict[node_id] = token.word
	   #      graph.add_node(self.unique_dict[token.word], label=token.word)
	        
	   #      if token.head >= 0:
				# src_id = token.head
				# graph.add_edge(
				#   src_id,
				#   node_id,
				#   label=token.label,
				#   key="parse_{}_{}".format(node_id, src_id))
	    return graph

	def gen_data_one(self, sentence):

		
		# sent = [word for word in sentence.split()]

		# for word in sent:
		# 	if word not in self.vocab:
		# 		self.vocab[word] = 0
		# 	else:
		# 		self.vocab[word] += 1

		# self.unique_words = self.vocab.keys()

		# for i, word in enumerate(self.unique_words):
		# 	self.unique_dict[word] = i

		return

##################################### helper functions ##############################################

def test_tree_build(s):
	# sanity check for one sentence
	#s = 'you cannot believe in god until you believe in yourself'
	parse_tree = tree_gen.get_tree(s)
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



if __name__ == '__main__':
	
	argList = [None]
	args = Args(argList)
	G = GraphBuilder()
	# if option is default then
	#sents = np.load(args.input)
	while True:
		G.gen_data(sents=None)
		G.gen_giant_graph()
		break
		# if self.done:
		# 	break
	##G.plot_graph(G.giant_graph)
	args.graph = G.giant_graph
	args.meta_paths = G.meta_paths
	model = model_maker(args)
	visualize(model, './output/', G.giant_graph)
	
