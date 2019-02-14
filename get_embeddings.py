import tensorflow as tf
import tensorflow_hub as hub
import nltk
import numpy as np

class Model:
	def __init__(self):
		self.max_steps = 50
		self.embeddings = None
		self.tokens_input = None
		self.tokens_length = None

		self.get_embeddings()

	def get_embeddings(self):

		self.tokens_input = tf.placeholder(tf.string, shape=[None, self.max_steps], name='data')
		self.tokens_length = tf.placeholder(tf.int32, shape=[None], name='data')

		elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable=False)

		self.embeddings = elmo(
			inputs={
				"tokens": self.tokens_input,
				"sequence_len": self.tokens_length
				},
				signature="tokens",
				as_dict=True)["elmo"]


	def step(self, tokens_input, tokens_length):

		feed_dict = dict()

		feed_dict[self.tokens_length] = tokens_length
		feed_dict[self.tokens_input] = tokens_input

		return (self.embeddings, self.tokens_input, self.tokens_length), feed_dict