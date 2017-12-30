"""
Author: Shubham Bhardwaj
Github: shubham0704
"""

import re
import nltk
import string
import logging
import numpy as np
from unidecode import unidecode
from memory_profiler import profile
from nltk.tokenize import WordPunctTokenizer

word_punct_tokenizer = WordPunctTokenizer()

stp_words = ["lrb", "rrb", "sjg","``", "''", ',']


f = open("./data/wiki_complete_dump_2008.txt.tokenized")
tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")




def read_in_chunks(file_object, chunk_size=10240):
    """Lazy function (generator) to read a file piece by piece.
    Default chunk size: 1k."""
    while True:
        data = file_object.read(chunk_size)
        if not data:
            break
        yield data



def lazy_load(file_object=f):
	"""returns 2-D sents list with each sent in word tokenized format"""

	try:
		chunk = next(read_in_chunks(f))
	except Exception as e:
		logging.warning("File read fully: ", e)
		f.close()
		return None
	
	sents = tokenizer.tokenize(chunk.decode('utf-8'))

	tok_sents = []
	sentences = []
	table = string.maketrans("","")
	for sent in sents:
	    tok_sent = [word for word in word_punct_tokenizer.tokenize(unidecode(sent)) if word not in string.punctuation and word.isalnum() and len(word)>1 and word not in stp_words and not(re.match('^[\'-]', word))]
	    if tok_sent:
	        tok_sents.append(tok_sent)
	        sentence = " ".join(tok_sent)
	        sentence = sentence.translate(table, string.punctuation)
	        sentences.append(sentence)

	return tok_sents, sentences

if __name__ == '__main__':
	x,y = lazy_load()
