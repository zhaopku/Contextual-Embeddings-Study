import os
from collections import defaultdict
import nltk
from tqdm import tqdm
import pickle as p
import random

max_length = 50

class TextData:
	def __init__(self):
		self.path = 'snli_1.0/text.txt'
		self.word_freq = defaultdict(int)

		self.sents = []
		self.length = []
		self.read_sents()

		self.word_descending = [(w, n) for w, n in self.word_freq.items()]
		self.word_descending = sorted(self.word_descending, key=lambda x: (-x[1], x[0]))

		self.filtered_sents = []
		self.filtered_length = []
		self.filtering(k=200)

	def filtering(self, k):
		self.top_k_words = set([x for (x, y) in self.word_descending[200:200+k]])

		for idx, sent in enumerate(tqdm(self.sents, desc='filtering')):
			sent_set = set(sent)
			if len(sent_set.intersection(self.top_k_words)) > 0:
				# if sent in self.filtered_sents:
				# 	continue
				self.filtered_sents.append(sent)
				self.filtered_length.append(self.length[idx])

		print(len(self.filtered_sents))

	def read_sents(self):
		with open(self.path, 'r') as file:
			lines = file.readlines()
			random.shuffle(lines)

			for line in tqdm(lines):
				words = nltk.word_tokenize(line.lower().strip())

				if len(words) > max_length:
					continue

				for word in words:
					self.word_freq[word] += 1

				self.length.append(len(words))

				while len(words) < max_length:
					words.append('pad')

				words = words[:max_length]

				self.sents.append(words)


if __name__ == '__main__':

	if os.path.exists('snli.pkl'):
		with open('snli.pkl', 'rb') as file:
			text_data = p.load(file)
	else:
		text_data = TextData()
		with open('snli.pkl', 'wb') as file:
			p.dump(text_data, file)

	print()
