import os
import random
import string
import nltk.chunk
from nltk.corpus import conll2002
from nltk.stem import SnowballStemmer

def generate_feature(data, prev, next, position, sentence_length, stemmer):
	feature = {}
	word, pos, tag = data
	prev_word, prev_pos, prev_tag = prev
	next_word, next_pos, next_tag = next
	stem = stemmer.stem(word)
	prev_stem = stemmer.stem(prev_word)
	next_stem = stemmer.stem(next_word)
	#feature['UPPER'] = word.upper()
	#feature['LOWER'] = word.lower()
	feature['WORD'] = word
	feature['POS'] = pos
	feature['STEM'] = stem
	feature['PREVWORD'] = prev_word
	feature['PREVPOS'] = prev_pos
	feature['PREVTAG'] = prev_tag
	feature['PREVSTEM'] = prev_stem
	feature['NEXTWORD'] = next_word
	feature['NEXTPOS'] = next_pos
	feature['NEXTSTEM'] = next_word
	feature['POS+PREVTAG'] = pos + prev_tag
	feature['PREVTAG'] = prev_tag
	feature['PREVWORD+WORD'] = prev_word + word
	feature['WORD+NEXTWORD'] = word + next_word
	feature['PREVWORD+WORD+NEXTWORD'] = prev_word + word + next_word
	#Structure
	feature['PUNCTUATION'] = False
	feature['DIGIT'] = False
	for ch in word:
		if ch in string.punctuation:
			feature['PUNCTUATION'] = True
		if ch.isdigit() and not word.isdigit():
			feature['DIGIT'] = True
	#Shape
	feature['ISUPPER'] = word.isupper()
	feature['MIX'] = not word.isupper() and not word.islower() and not word.istitle()
	#Prefix
	feature['FIRST6'] = ''
	if len(word) >= 6:
		feature['FIRST6'] = word[ : 6]
	#Suffix
	feature['LAST2'] = ''
	if len(word) >= 2:
		feature['LAST2'] = word[-2 : ]
	feature['LAST1'] = ''
	if len(word) >= 1:
		feature['LAST1'] = word[-1 : ]
	feature['PREVPOS+POS+NEXTPOS'] = prev_pos + pos + next_pos
	feature['TITLE'] = word.istitle()
	feature['PREVTITLE+NEXTTITLE+POS'] = str(prev_word.istitle() and next_word.istitle()) + pos
	#feature['TITLE+NEXTTITLE'] = word.istitle() and next_word.istitle()
	#feature['NEXTTITLE'] = next_word.istitle()
	#feature['TITLE|UPPER'] = word.istitle() or word.isupper()
	feature['WORDLEN'] = len(word)
	return feature

#Predicting label for test set using Viterbi search
def predict(maxent, test_set):
	labels, preds = [], []
	viterbi = {}
	previous_probs = {}
	categories = maxent.labels()
	for category in categories:
		viterbi[category] = ['']
		previous_probs[category] = 1
	prev_pred = 'O'
	for sample in test_set:
		feature, label = sample
		#Use previous prediction as previous tag feature for test set
		feature['PREVTAG'] = prev_pred
		probs = maxent.prob_classify(feature)
		prev_pred = probs.max()
		probs_val = {}
		norm = 0
		for category in viterbi:
			max_prob = 0
			max_category = ''
			prob = probs.prob(category)
			for prev_category in previous_probs:
				cur_prob = previous_probs[prev_category] * prob
				if cur_prob > max_prob:
					max_prob = cur_prob
					max_category = prev_category
			viterbi[category].append(max_category)
			prob *= previous_probs[max_category]
			probs_val[category] = prob
		if len(set(probs_val.values())) > 1:
			for category in previous_probs:
				previous_probs[category] = probs_val[category]
				norm += previous_probs[category]
			for category in previous_probs:
				previous_probs[category] /= norm
		labels.append(label)
	max_prob = 0
	max_category = ''
	for category in previous_probs:
		if previous_probs[category] > max_prob:
			max_prob = previous_probs[category]
			max_category = category
	last = len(viterbi['O']) - 1
	while last > 0:
		preds.append(max_category)
		max_category = viterbi[max_category][last]
		last -= 1
	return labels, preds[::-1]

#Calculating recall, precision and F1-score for every category and overall performance
def f_measure(labels, preds):
	prev_label = 'O'
	prev_pred = 'O'
	total = {
			'PER' : 0,
			'ORG' : 0,
			'LOC' : 0,
			'MISC': 0,
			'OVERALL' : 0
			}
	predicted = {
			'PER' : 0,
			'ORG' : 0,
			'LOC' : 0,
			'MISC': 0,
			'OVERALL' : 0
			}
	correct = {
			'PER' : 0,
			'ORG' : 0,
			'LOC' : 0,
			'MISC': 0,
			'OVERALL' : 0
			}
	f1 = {
			'PER' : 0,
			'ORG' : 0,
			'LOC' : 0,
			'MISC' : 0,
			'OVERALL' : 0
			}
	recall = {
			'PER' : 0,
			'ORG' : 0,
			'LOC' : 0,
			'MISC' : 0,
			'OVERALL' : 0
			}
	precision = {
			'PER' : 0,
			'ORG' : 0,
			'LOC' : 0,
			'MISC' : 0,
			'OVERALL' : 0
			}
	match = 0
	in_chunk = False
	category_label = 'O'
	category_pred = 'O'
	n = len(labels)
	for i in range(n):
		label = labels[i]
		pred = preds[i]
		if label == pred:
			match += 1
		if len(label) > 1:
			category_label = label[2 : ]
		if len(pred) > 1:
			category_pred = pred[2 : ]
		if label[0] == 'B':
			total[category_label] += 1
			total['OVERALL'] += 1
			in_chunk = True
		if pred[0] == 'B':
			predicted[category_pred] += 1
			predicted['OVERALL'] += 1
		if in_chunk:
			if pred != label or label[0] == 'O':
				in_chunk = False
			if pred == label and label[0] == 'O':
				correct[category_label] += 1
				correct['OVERALL'] += 1
			if not in_chunk:
				category_label = 'O'
				category_pred = 'O'
	for category in total:
		recall[category] = correct[category] * 1.0 / total[category]
		precision[category] = correct[category] * 1.0 / predicted[category]
		f1[category] = 2 * recall[category] * precision[category] / (recall[category] + precision[category])
	accuracy = match * 1.0 / n
	return f1, total, predicted, correct, accuracy, recall, precision

#Training and testing procedure
print 'Select language: spanish/dutch'
language = raw_input()
print 'Select test set: dev/test'
mode = raw_input()
train_file, test_file = '', ''
if language == 'spanish':
	train_file = 'esp.train'
	if mode == 'dev':
		test_file = 'esp.testa'
	else:
		test_file = 'esp.testb'
elif language == 'dutch':
	train_file = 'ned.train'
	if mode == 'dev':
		test_file = 'ned.testa'
	else:
		test_file = 'ned.testb'
stemmer = SnowballStemmer(language)
chunked = [nltk.chunk.tree2conlltags(tree) for tree in conll2002.chunked_sents(fileids = train_file)]
print 'Generating training set'
train_set = []
for chunk in chunked:
	for i in range(len(chunk)):
		prev = ('', '', '')
		if i > 0:
			prev = chunk[i - 1]
		next = ('', '', '')
		if i < len(chunk) - 1:
			next = chunk[i + 1]
		data = chunk[i]
		feature = generate_feature(data, prev, next, i, len(chunk), stemmer)
		train_set.append((feature, data[-1]))
chunked = [nltk.chunk.tree2conlltags(tree) for tree in conll2002.chunked_sents(fileids = test_file)]
print 'Generating test set'
test_set = []
for chunk in chunked:
	for i in range(len(chunk)):
		prev = ('', '', '')
		if i > 0:
			prev = chunk[i - 1]
		next = ('', '', '')
		if i < len(chunk) - 1:
			next = chunk[i + 1]
		data = chunk[i]
		feature = generate_feature(data, prev, next, i, len(chunk), stemmer)
		test_set.append((feature, data[-1]))
dir = os.getcwd()
nltk.config_megam(dir + '/megam_i686.opt')
print 'Training'
maxent = nltk.MaxentClassifier.train(train_set, algorithm='megam')
print 'Testing'
labels, preds = predict(maxent, test_set)
print 'Statistic'
f1, total, predicted, correct, accuracy, recall, precision = f_measure(labels, preds)
print 'Total: ', total
print 'Predicted: ', predicted
print 'Correct: ', correct
print 'Recall: ', recall
print 'Precision: ', precision
print 'F1 score: ', f1
print 'Tag accuracy: ', accuracy
