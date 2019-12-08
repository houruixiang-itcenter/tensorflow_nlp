import numpy as np
import tensorflow as tf
import collections
import random
import math
import ssl
import os

ssl._create_default_https_context = ssl._create_unverified_context

url = 'https://www.cs.cmu.edu/~spok/grimmtmp/'
num_files = 100
dir_name = 'stories'

documents = []

chars = []
data_list = []
count = []
dictionary = dict({'UNK': 0})
reverse_dictionary = dict()
vocabulary_size = 0

if not os.path.exists(dir_name):
	os.mkdir(dir_name)

data_indices = None
data_list = None
reverse_dictionary = None
embedding_size = None
vocabulary_size = None
num_files = None


def define_data_and_hyperparameters(_num_files, _data_list, _reverse_dictionary, _emb_size, _vocab_size):
	global num_files, data_indices, data_list, reverse_dictionary
	global embedding_size, vocabulary_size
	
	num_files = _num_files
	data_indices = [0 for _ in range(num_files)]
	data_list = _data_list
	reverse_dictionary = _reverse_dictionary
	embedding_size = _emb_size
	vocabulary_size = _vocab_size


def maybe_download(filename):
	"""Download a file if not present"""
	print('Downloading file: ', dir_name + os.sep + filename)
	
	if not os.path.exists(dir_name + os.sep + filename):
		filename, _ = urlretrieve(url + filename, dir_name + os.sep + filename)
	else:
		print('File ', filename, ' already exists.')
	
	return filename


def read_data(filename):
	with open(filename) as f:
		data = tf.compat.as_str(f.read())
		# make all the words lower case  words 转换为小写
		data = data.lower()
		data = list(data)
	return data


def build_dataset(documents):
	'''

	:param documents:
	:return:
	count list - [[word,fred]]
	dictionary dict - {word:id}
	reverse_dictionary - {id,word}
	'''
	chars = []
	# This is going to be a list of lists
	# Where the outer list denote each document
	# and the inner lists denote words in a given document
	data_list = []
	
	for d in documents:
		chars.extend(d)
	print('%d Characters found.' % len(chars))
	count = []
	# Get the bigram sorted by their frequency (Highest comes first)
	count.extend(collections.Counter(chars).most_common())
	
	# Create an ID for each bigram by giving the current length of the dictionary
	# And adding that item to the dictionary
	# Start with 'UNK' that is assigned to too rare words
	dictionary = dict({'UNK': 0})
	for char, c in count:
		# Only add a bigram to dictionary if its frequency is more than 10
		if c > 10:
			dictionary[char] = len(dictionary)
	
	unk_count = 0
	# Traverse through all the text we have
	# to replace each string word with the ID of the word
	
	for d in documents:
		data = list()
		for char in d:
			# If word is in the dictionary use the word ID,
			# else use the ID of the special token "UNK"
			if char in dictionary:
				index = dictionary[char]
			else:
				index = dictionary['UNK']
				unk_count += 1
			data.append(index)
		
		data_list.append(data)
	
	reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
	return data_list, count, dictionary, reverse_dictionary


def generate_batch_for_word2vec(data_list, doc_id, batch_size, window_size):
	# window_size is the amount of words we're looking at from each side of a given word
	# creates a single batch
	# doc_id is the ID of the story we want to extract a batch from
	
	# data_indices[doc_id] is updated by 1 everytime we read a set of data point
	# from the document identified by doc_id
	global data_indices
	
	# span defines the total window size, where
	# data we consider at an instance looks as follows.
	# [ skip_window target skip_window ]
	# e.g if skip_window = 2 then span = 5
	span = 2 * window_size + 1
	
	# two numpy arras to hold target words (batch)
	# and context words (labels)
	# Note that batch has span-1=2*window_size columns
	batch = np.ndarray(shape=(batch_size, span - 1), dtype=np.int32)
	labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
	
	# The buffer holds the data contained within the span
	buffer = collections.deque(maxlen=span)
	
	# Fill the buffer and update the data_index
	for _ in range(span):
		buffer.append(data_list[doc_id][data_indices[doc_id]])
		data_indices[doc_id] = (data_indices[doc_id] + 1) % len(data_list[doc_id])
	
	# Here we do the batch reading
	# We iterate through each batch index
	# For each batch index, we iterate through span elements
	# to fill in the columns of batch array
	for i in range(batch_size):
		target = window_size  # target label at the center of the buffer
		target_to_avoid = [window_size]  # we only need to know the words around a given word, not the word itself
		
		# add selected target to avoid_list for next time
		col_idx = 0
		for j in range(span):
			# ignore the target word when creating the batch
			if j == span // 2:
				continue
			batch[i, col_idx] = buffer[j]
			col_idx += 1
		labels[i, 0] = buffer[target]
		
		# Everytime we read a data point,
		# we need to move the span by 1
		# to update the span
		buffer.append(data_list[doc_id][data_indices[doc_id]])
		data_indices[doc_id] = (data_indices[doc_id] + 1) % len(data_list[doc_id])
	
	assert batch.shape[0] == batch_size and batch.shape[1] == span - 1
	return batch, labels


def print_some_batches():
	global num_files, data_list, reverse_dictionary
	
	for window_size in [1, 2]:
		data_indices = [0 for _ in range(num_files)]
		batch, labels = generate_batch_for_word2vec(data_list, doc_id=0, batch_size=8, window_size=window_size)
		print('\nwith window_size = %d:' % (window_size))
		print('    batch:', [[reverse_dictionary[bii] for bii in bi] for bi in batch])
		print('    labels:', [reverse_dictionary[li] for li in labels.reshape(8)])


batch_size, embedding_size, window_size = None, None, None
valid_size, valid_window, valid_examples = None, None, None
num_sampled = None

train_dataset, train_labels = None, None
valid_dataset = None

softmax_weights, softmax_biases = None, None

loss, optimizer, similarity, normalized_embeddings,normal,valid_embeddings = None, None, None, None,None,None


def define_word2vec_tensorflow():
	global batch_size, embedding_size, window_size
	global valid_size, valid_window, valid_examples
	global num_sampled
	global train_dataset, train_labels
	global valid_dataset
	global softmax_weights, softmax_biases
	global loss, optimizer, similarity
	global vocabulary_size, embedding_size
	global normalized_embeddings,normal,valid_embeddings
	
	batch_size = 128  # Data points in a single batch
	
	# How many words to consider left and right.
	# Skip gram by design does not require to have all the context words in a given step
	# However, for CBOW that's a requirement, so we limit the window size
	window_size = 3
	
	# We pick a random validation set to sample nearest neighbors
	valid_size = 16  # Random set of words to evaluate similarity on.
	# We sample valid datapoints randomly from a large window without always being deterministic
	valid_window = 50
	
	# When selecting valid examples, we select some of the most frequent words as well as
	# some moderately rare words as well
	valid_examples = np.array(random.sample(range(valid_window), valid_size))
	valid_examples = np.append(valid_examples, random.sample(range(400, 400 + valid_window), valid_size), axis=0)
	
	num_sampled = 32  # Number of negative examples to sample.
	
	tf.reset_default_graph()
	
	# Training input data (target word IDs). Note that it has 2*window_size columns
	train_dataset = tf.placeholder(tf.int32, shape=[batch_size, 2 * window_size])
	# Training input label data (context word IDs)
	train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
	# Validation input data, we don't need a placeholder
	# as we have already defined the IDs of the words selected
	# as validation data
	valid_dataset = tf.constant(valid_examples, dtype=tf.int32)
	
	# Variables.
	
	# Embedding layer, contains the word embeddings
	embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0, dtype=tf.float32))
	
	# Softmax Weights and Biases
	softmax_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size],
	                                                  stddev=0.5 / math.sqrt(embedding_size), dtype=tf.float32))
	softmax_biases = tf.Variable(tf.random_uniform([vocabulary_size], 0.0, 0.01))
	
	# Model.
	# Look up embeddings for a batch of inputs.
	# Here we do embedding lookups for each column in the input placeholder
	# and then average them to produce an embedding_size word vector
	stacked_embedings = None
	print('Defining %d embedding lookups representing each word in the context' % (2 * window_size))
	for i in range(2 * window_size):
		embedding_i = tf.nn.embedding_lookup(embeddings, train_dataset[:, i])
		x_size, y_size = embedding_i.get_shape().as_list()
		if stacked_embedings is None:
			stacked_embedings = tf.reshape(embedding_i, [x_size, y_size, 1])
		else:
			stacked_embedings = tf.concat(axis=2,
			                              values=[stacked_embedings, tf.reshape(embedding_i, [x_size, y_size, 1])])
	
	assert stacked_embedings.get_shape().as_list()[2] == 2 * window_size
	print("Stacked embedding size: %s" % stacked_embedings.get_shape().as_list())
	mean_embeddings = tf.reduce_mean(stacked_embedings, 2, keepdims=False)
	print("Reduced mean embedding size: %s" % mean_embeddings.get_shape().as_list())
	
	# Compute the softmax loss, using a sample of the negative labels each time.
	# inputs are embeddings of the train words
	# with this loss we optimize weights, biases, embeddings
	loss = tf.reduce_mean(
		tf.nn.sampled_softmax_loss(weights=softmax_weights, biases=softmax_biases, inputs=mean_embeddings,
		                           labels=train_labels, num_sampled=num_sampled, num_classes=vocabulary_size))
	# AdamOptimizer.
	optimizer = tf.train.AdamOptimizer(0.0005).minimize(loss)
	
	# Compute the similarity between minibatch examples and all embeddings.
	# We use the cosine distance:
	norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keepdims=True))
	normalized_embeddings = embeddings / norm
	valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
	similarity = tf.matmul(valid_embeddings, tf.transpose(normalized_embeddings))


def run_word2vec():
	global batch_size, embedding_size, window_size
	global valid_size, valid_window, valid_examples
	global num_sampled
	global train_dataset, train_labels
	global valid_dataset
	global softmax_weights, softmax_biases
	global loss, optimizer, similarity, normalized_embeddings
	global data_list, num_files, reverse_dictionary
	global vocabulary_size, embedding_size
	
	num_steps = 10
	steps_per_doc = 100
	
	session = tf.InteractiveSession()
	
	# Initialize the variables in the graph
	tf.global_variables_initializer().run()
	print('Initialized')
	
	average_loss = 0
	
	for step in range(num_steps):
		
		# Iterate through the documents in a random order
		for doc_id in np.random.permutation(num_files):
			for doc_step in range(steps_per_doc):
				# Generate a single batch of data from a document
				batch_data, batch_labels = generate_batch_for_word2vec(data_list, doc_id, batch_size, window_size)
				
				# Populate the feed_dict and run the optimizer (minimize loss)
				# and compute the loss
				feed_dict = {train_dataset: batch_data, train_labels: batch_labels}
				_, l = session.run([optimizer, loss], feed_dict=feed_dict)
				
				average_loss += l
		
		if (step + 1) % 1 == 0:
			if step > 0:
				# compute average loss
				average_loss = average_loss / (doc_id * steps_per_doc)
			
			print('Average loss at step %d: %f' % (step + 1, average_loss))
			average_loss = 0  # reset average loss
		
		# Evaluating validation set word similarities
		if (step + 1) % 5 == 0:
			sim = similarity.eval()
			
			# Here we compute the top_k closest words for a given validation word
			# in terms of the cosine distance
			# We do this for all the words in the validation set
			# Note: This is an expensive step
			for i in range(valid_size):
				valid_word = reverse_dictionary[valid_examples[i]]
				top_k = 4  # number of nearest neighbors
				nearest = (-sim[i, :]).argsort()[1:top_k + 1]
				log = 'Nearest to %s:' % valid_word
				for k in range(top_k):
					close_word = reverse_dictionary[nearest[k]]
					log = '%s %s,' % (log, close_word)
				print(log)
	cbow_final_embeddings = normalized_embeddings.eval()
	
	# We save the embeddings as embeddings.npy
	np.save('embeddings', cbow_final_embeddings)


def main():
	# 下载格林通话前100篇
	filenames = [format(i, '03d') + '.txt' for i in range(1, 101)]
	
	for fn in filenames:
		maybe_download(fn)
	num_files = 100
	global documents
	for i in range(num_files):
		print('\nProcessing file %s' % os.path.join(dir_name, filenames[i]))
		chars = read_data(os.path.join(dir_name, filenames[i]))
		two_grams = [''.join(chars[ch_i:ch_i + 2]) for ch_i in range(0, len(chars) - 2, 2)]
		documents.append(two_grams)
		print('Data size (Characters) (Document %d) %d' % (i, len(two_grams)))
		print('Sample string (Document %d) %s' % (i, two_grams[:50]))
	# 引用全局变量
	global data_list, count, dictionary, reverse_dictionary, vocabulary_size
	data_list, count, dictionary, reverse_dictionary = build_dataset(documents)
	vocabulary_size = len(dictionary)
	
	# 加载word2vec
	embedding_size = 128
	define_data_and_hyperparameters(
		num_files, data_list, reverse_dictionary, embedding_size, vocabulary_size)
	# word2vec.print_some_batches()
	define_word2vec_tensorflow()
	
	run_word2vec()


if __name__ == '__main__':
	main()
