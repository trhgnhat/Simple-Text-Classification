from gensim.models import Word2Vec
import underthesea as uts
import os
import urllib.request
import numpy as np
from bs4 import BeautifulSoup
import keras
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense, Bidirectional

def clean_script(html):
	"""
			Clean html tags, scripts and css code
			:param html: input html content
			:return: cleaned text
    """
	soup = BeautifulSoup(html)
	for script in soup(["script", "style"]):
		script.extract()
	# get text
	text = soup.get_text()
	# break into lines and remove leading and trailing space on each
	lines = (line.strip() for line in text.splitlines())
	# break multi-headlines into a line each
	chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
	# drop blank lines
	text = '\n'.join(chunk for chunk in chunks if chunk)
	return text


def download_html(url_path, output_path, should_clean=True):
	"""
			Download html content from url
			:param url_path: input url
			:param output_path: path to output file
			:param should_clean: should clean or not
			:return: cleaned text
	"""
	with urllib.request.urlopen(url_path) as response:
		html = response.read()
		if should_clean:
			text = clean_script(html)
		else:
			text = html
	with open(output_path, 'w', encoding='utf8') as fw:
		fw.write(text)
	return text


def load_data(path_list=[], n_class=3):
	X = None
	y = None
	for i, data_path in enumerate(path_list):
		print(data_path)
		with open(data_path, 'r', encoding='utf8') as fr:
			sentences_ = fr.readlines()
			sentences_ = [sent.strip() for sent in sentences_ if len(sent.strip()) > 0]
		print(sentences_)
		label_vec = [0.0 for _ in range(0, n_class)]
		label_vec[i] = 1.0
		labels_ = [label_vec for _ in range(0, len(sentences_))]
		if X is None:
			X = sentences_
			y = labels_
		else:
			X += sentences_
			y += labels_
	return X, y


def tokenize(sentence):
	return uts.word_tokenize(sentence)


def tokenize_sentences(sentences):
	"""
			Tokenize or word segment sentences
			:param sentences: input sentences
			:return: tokenized sentence
	"""
	tokens_list = []
	max_length = -1
	for sent in sentences:
		tokens = tokenize(sent)
		tokens_list.append(tokens)
		if len(tokens) > max_length:
			max_length = len(tokens)

	return tokens_list, max_length


def word_embed_sentences(sentences, word2vec, max_length=20):
	"""
			Helper method to convert word to vector
			:param sentences: input sentences in list of strings format
			:param max_length: max length of sentence you want to keep, pad more or cut off
			:return: embedded sentences as a 3D-array
	"""
	embed_sentences = []
	word_dim = word2vec[word2vec.index2word[0]].shape[0]
	for sent in sentences:
		embed_sent = []
		for word in sent:
			if word.lower() in word2vec:
				embed_sent.append(word2vec[word.lower()])
			else:
				embed_sent.append(np.zeros(shape=(word_dim,), dtype=float))

		if len(embed_sent) > max_length:
			embed_sent = embed_sent[:max_length]
		elif len(embed_sent) < max_length:
			embed_sent = np.concatenate(
					(embed_sent,
					np.zeros(shape=(max_length - len(embed_sent), word_dim), dtype=float)),
					axis=0)

		embed_sentences.append(embed_sent)

	return embed_sentences


def build_model(input_dim, n_class=3):
	"""
	Overwrite build model using Bidirectional Layer
	:param input_dim: input dimension
	:return: Keras model
	"""
	model = Sequential()

	model.add(Bidirectional(LSTM(32, return_sequences=True), input_shape=input_dim))
	model.add(Dropout(0.1))
	model.add(Bidirectional(LSTM(16)))
	model.add(Dense(n_class, activation="softmax"))

	model.compile(loss=keras.losses.categorical_crossentropy,
	              optimizer=keras.optimizers.Adadelta(),
	              metrics=['accuracy'])
	return model


W2V_PRETRAINED_MODEL_PATH = os.getcwd() + '/models/pretrained_word2vec.bin'
DATA_PATH = os.getcwd() + '/data'


if __name__ == "__main__":
	# # Download data
	# url_path = "https://dantri.com.vn/su-kien/anh-huong-bao-so-6-dem-nay-mot-so-tinh-dong-bac-bo-co-gio-giat-manh-20180916151250555.htm"
	# output_path = os.getcwd() + "/data/word_embedding/real/html/html_data2.txt"
	# download_html(url_path, output_path)
	#
	# # Word Embedding train
	# file_path = output_path
	# sentences = []
	# batch_sentences = []
	# with open(file_path, 'r', encoding='utf8') as fr:
	# 	lines = fr.readlines()
	# 	for line in lines:
	# 		sent = tokenize(line.strip())
	# 		batch_sentences.append(sent)
	# if sentences is None:
	# 	sentences = batch_sentences
	# else:
	# 	sentences += batch_sentences
	# w2v_model = Word2Vec(sentences, size=100, window=5, min_count=1, workers=4)

	"""
		Due to the lack of data, the above Word2Vec model are weak to form vectors.
		Therefore, pretrained model are used to ensure the best prediction.
	"""

	#use pretrained Word2Vec model
	word2vec_model = Word2Vec.load(W2V_PRETRAINED_MODEL_PATH)
	# Create data
	X, y = load_data([DATA_PATH + '/positive.txt',
	                  DATA_PATH + '/negative.txt'],
	                 n_class=2)
	max_length = 10
	X = [tokenize(sent) for sent in X]
	X = word_embed_sentences(X, word2vec_model.wv, max_length)
	X = np.array(X)
	y = np.array(y)

	# Build model
	model = build_model(input_dim=(X.shape[1], X.shape[2]), n_class=2)
	n_epochs = 10
	batch_size = 6
	model.fit(X, y, batch_size=batch_size, epochs=n_epochs)

	# Prediction test
	label_dict = {0: 'tích cực', 1: 'tiêu cực'}
	test_sentences = ['Dở thế', 'Hay thế', 'phim chán thật', 'nhảm quá', 'rất hay']
	X_test = [tokenize(sent) for sent in test_sentences]
	X_test = word_embed_sentences(X_test , word2vec_model.wv, max_length=max_length)
	y = model.predict(np.array(X_test))
	print(y)
	y = np.argmax(y, axis=1)  # get the max_element's index of all row
	labels = []
	for lab_ in y:
		labels.append(label_dict[lab_])
	print(labels)