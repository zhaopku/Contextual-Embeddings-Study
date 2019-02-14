import pickle as p
import os
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.style
import matplotlib
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--n', type=int, default=2, help='only valid when interactive is true')
parser.add_argument('--word', type=str, default='fire')
parser.add_argument('--interactive', action='store_true')
args = parser.parse_args()

root_dir = 'images'
n_clusters = [2, 3, 4]

def plot_results(labels, reduced_embeds, save_dir, n, word):
	matplotlib.rcParams['figure.figsize'] = [16.0, 15.0]

	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	cmaps = [
		'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
		'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
		'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']

	for idx in range(n):
		ax.scatter(reduced_embeds[labels==idx,0], reduced_embeds[labels==idx,1], reduced_embeds[labels==idx,2],
		           cmap=cmaps[idx], label = 'cluster {}'.format(idx))

	ax.legend(loc="center left", bbox_to_anchor=(1,0.5))
	plt.title(word)

	plt.savefig(save_dir)

	if args.interactive:
		plt.show()
		plt.pause(0.1)

		input('---------- Press enter to quit. ----------')
		plt.close()

	else:
		plt.savefig(save_dir)

def save_clusters(word, embeds, n):
	save_dir = os.path.join(root_dir, word)
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)

	save_dir = os.path.join(save_dir, str(n)+'.png')

	kmeans = KMeans(n_clusters=n, n_init=8, n_jobs=-1).fit(embeds)

	pca = PCA(n_components=3)

	reduced_embeds = pca.fit_transform(embeds)

	plot_results(labels=kmeans.labels_, reduced_embeds=reduced_embeds, save_dir=save_dir, n=n, word=word)

with open('words2embed.pkl', 'rb') as file:
	word2embed = p.load(file)

	if args.interactive:
		if not args.word in word2embed.keys():
			print('{} not exists in dict'.format(args.word))
			exit(-1)

		save_clusters(args.word, word2embed[args.word], args.n)
		exit(0)

	for word, embeds in tqdm(word2embed.items(), desc='processing words'):
		for n in n_clusters:
			save_clusters(word, embeds, n)