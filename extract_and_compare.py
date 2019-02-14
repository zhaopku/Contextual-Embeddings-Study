import tensorflow as tf
import tensorflow_hub as hub
import nltk
import numpy as np
from TextData import TextData
import pickle as p
import os
from get_embeddings import Model
from collections import defaultdict
from tqdm import tqdm

def padding(seqs, length):
	new_seqs = []

	for seq in seqs:
		while len(seq) < length:
			seq.append(seq[-1])
		new_seqs.append(seq)

	return new_seqs

def compute_distance(xs, ys):
	d = []
	for i, x in enumerate(xs):
		for j, y in enumerate(ys):
			if j <= i:
				continue
			d.append(np.linalg.norm(x-y))

	return d

def find_index_top_k(sent, top_k_words):
	pass

if __name__ == '__main__':
	if os.path.exists('snli.pkl'):
		with open('snli.pkl', 'rb') as file:
			text_data = p.load(file)
	else:
		text_data = TextData()
		with open('snli.pkl', 'wb') as file:
			p.dump(text_data, file)

	assert len(text_data.filtered_length) == len(text_data.filtered_sents)

	batch_size = 50

	model = Model()

	init = tf.global_variables_initializer()

	words2embeds = defaultdict(list)

	with tf.Session() as sess:
		sess.run(init)

		num_batch = len(text_data.filtered_length)//batch_size

		for i in tqdm(range(2)):
			tokens_input = text_data.filtered_sents[i*batch_size:(i+1)*batch_size]
			tokens_length = text_data.filtered_length[i*batch_size:(i+1)*batch_size]

			ops, feed_dict = model.step(tokens_input=tokens_input, tokens_length=tokens_length)

			embeddings, tokens_input_, tokens_length_ = sess.run(ops, feed_dict=feed_dict)

			for m in range(len(tokens_input)):
				# max sent length is 50
				for n in range(50):
					if tokens_input[m][n] in text_data.top_k_words:
						if len(words2embeds[tokens_input[m][n]]) <= 300 and tokens_input[m][n] in text_data.top_k_words:
							words2embeds[tokens_input[m][n]].append(embeddings[m][n])



	with open('words2embed2.pkl', 'wb') as file:
		p.dump(words2embeds, file)
