import tensorflow as tf
import tensorflow_hub as hub
import nltk
import numpy as np


def padding(seqs, length):
	new_seqs = []

	for seq in seqs:
		while len(seq) < length:
			seq.append(seq[-1])
		new_seqs.append(seq)

	return new_seqs

elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable=False)

sents = ['There is a time to work and a time to play.',
         'Let\'s play a different game.',
         'You will have to play inside today.',
         'This is a play from Shakespeare.',
         'I love this radio play.',
         'I am performing a play in front of my parents.']

tokens_input = [nltk.word_tokenize(x) for x in sents]

tokens_length = [len(x) for x in tokens_input]

tokens_input = padding(tokens_input, np.max(tokens_length))


indices = [10, 2, 4, 3, 4, 4]

embeddings = elmo(
	inputs={
		"tokens": tokens_input,
		"sequence_len": tokens_length
		},
		signature="tokens",
		as_dict=True)["elmo"]


init = tf.global_variables_initializer()

def compute_distance(xs, ys):
	d = []
	for i, x in enumerate(xs):
		for j, y in enumerate(ys):
			if j <= i:
				continue
			d.append(np.linalg.norm(x-y))
	return d

with tf.Session() as sess:
	sess.run(init)

	embeds = sess.run(embeddings)

	play_embeddings = []

	for i, idx in enumerate(indices):
		play_embeddings.append(embeds[i, idx])

	verbs = play_embeddings[:3]
	nouns = play_embeddings[3:]

	verb_distance = compute_distance(verbs, verbs)

	nouns_distance = compute_distance(nouns, nouns)

	inter_distance = []

	for v in verbs:
		for n in nouns:
			inter_distance.append(np.linalg.norm(v-n))

	print()
