#coding:utf-8
import numpy
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
max_length = 0
word_embeddings = []
position_embeddings = []
VocabList = {}
RelationList = {}
PositionLimit = 30
LenLimit = 140
word_size = 0
dim = 0
l2_lambda = 0.02
def getPosition(position):
	if (position < -PositionLimit):
		return 0
	if (position > PositionLimit):
		return 2*PositionLimit
	return position+PositionLimit
def getWordVec(path):
	f = open(path + 'vec_min50.txt', "r")
	content = f.readline()
	word_size, dim = [int(i) for i in content.strip().split()]
	word_size = word_size+2
	VocabList['UNK'] = 0
	VocabList['BLANK'] = 1
	word_embeddings.append([0.0 for i in range(dim)])
	word_embeddings.append([0.0 for i in range(dim)])
	while(True):
		content = f.readline()
		if content == "":
			break
		values = content.strip().split()
		VocabList[values[0]] = len(VocabList)
		values = values[1:]
		values = [(float)(i) for i in values]
		word_embeddings.append(values)

def getContent(content, print_info):
	id1, id2, name1, name2, r, s = content.strip().split("\t")
	if not r in RelationList:
		RelationList[r] = len(RelationList)
	r = RelationList[r]
	s = s.split()
	global max_length
	if len(s) > max_length:
		max_length = len(s)
	if print_info:
		print ' '.join(s)
	unk = VocabList['UNK']
	blank = VocabList['BLANK']
	position = 0
	for i in range(0, len(s)):
		if s[i] == name2:
			position = i
		if not s[i] in VocabList:
			s[i] = unk
		else:
			s[i] = VocabList[s[i]]
	for i in range(len(s), LenLimit):
		s.append(blank)
	if len(s) > LenLimit:
		s = s[0:LenLimit]
	positions = []
	for i in range(0, len(s)):
		positions.append(getPosition(i - position))
	if print_info:
		print s
		print positions
	return name1, name2, r, s, positions

def readFromFile(path):
	x_train = []
	r_train = []
	y_train = []
	p_train = []
	bags_train = {}
	x_test = []
	r_test = []
	y_test = []
	p_test = []
	bags_test = {}
	getWordVec(path)
	#load train.txt
	f = open(path + "train.txt", "r")
	id = 0
	while (True):
		content = f.readline()
		if content == "":
			break
		name1, name2, r, s, positions = getContent(content, False)
		x_train.append(s)
		p_train.append(positions)
		r_train.append(r)
		key = name1+' '+name2
		if key not in bags_train:
			bags_train[key] = []
		bags_train[key].append(id)
		id += 1
	f.close()
	for i in r_train:
		y = []
		for j in range(0, len(RelationList)):
			y.append(0)
		y[i] = 1
		y_train.append(y)
	#load test.txt
	f = open(path + "test.txt", "r")
	id = 0
	while (True):
		content = f.readline()
		if content == "":
			break
		name1, name2, r, s, positions = getContent(content, False)
		x_test.append(s)
		p_test.append(positions)
		r_test.append(r)
		key = name1+' '+name2
		if key not in bags_test:
			bags_test[key] = []
		bags_test[key].append(id)
		id += 1
	f.close()
	for i in r_test:
		y = []
		for j in range(0, len(RelationList)):
			y.append(0)
		y[i] = 1
		y_test.append(y)
	print 'max length = ', max_length
	return x_train, p_train, y_train, r_train, bags_train, x_test, p_test, y_test, r_test, bags_test

def batch_iter(x_train, p_train, y_train, r_train, bags_train):
	"""
	Generates a batch iterator for a dataset.
	"""
	data_size = len(bags_train)
	keys = bags_train.keys()
	keys_list = []
	for key in keys:
		keys_list.append(key)
	keys_array = numpy.array(keys_list)
 	# Shuffle the data at each epoch
	shuffle_indices = numpy.random.permutation(numpy.arange(data_size))
	shuffled_keys = keys_array[shuffle_indices]
	for key in shuffled_keys:
		bags = bags_train[key]
		x_batch = []
		p_batch = []
		for index in bags:
			x_batch.append(x_train[index])
			p_batch.append(p_train[index])
		y_batch = [y_train[bags[0]]]
		r_batch = [[r_train[bags[0]]]]
		'''
		print 'x--------------------------------------'
		print x_batch
		print 'p--------------------------------------'
		print p_batch
		print 'r--------------------------------------'
		print r_batch
		print 'y--------------------------------------'
		print y_batch
		print 'end--------------------------------------'
		'''
		yield x_batch, p_batch, y_batch, r_batch
