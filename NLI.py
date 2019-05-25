import argparse
import gc
import logging
import os
import pickle
import re
import sys

import numpy as np
from scipy import sparse
from sklearn import linear_model
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

import countries_native_family
import en_function_words

logger = logging.getLogger()


def parse_args():
	parser = argparse.ArgumentParser(description='Native Language Identification', epilog='By\nMatan Kolath\nMerav Mazouz')
	parser.add_argument('-t', '--text', type=str, help='Path to the text chunks')
	parser.add_argument('-p', '--pos', type=str, help='Path to the pos chunks')
	parser.add_argument('-c', '--threads', type=int, default=2, help='Number of threads to use to train')
	parser.add_argument('-i', '--load-in', type=str, default=None, help='Load in sample from file and vocabulary')
	parser.add_argument('-w', '--write-in', type=str, default=None, help='Write in sample to file')
	parser.add_argument('-o', '--load-out', type=str, default=None, help='Load out of sample from folder')
	parser.add_argument('-z', '--write-out', type=str, default=None, help='Write out of sample to folder')
	parser.add_argument('-m', '--read-models', action="store_true", default=False, help='Read models from file')
	parser.add_argument('-f', '--features', action="append", help='What type of features to use, can be given multiple times for multiple features\nLegal values: bow, pos, char3, fw', required=True)
	return parser.parse_args()


def set_log():
	logging.basicConfig(stream=sys.stdout, format='%(asctime)s %(message)s', level=logging.INFO)


def get_chunks_folders(path):
	countries_paths = {}
	for country_path in os.listdir(path):
		users_paths = {}
		country_full_path = os.path.join(path, country_path)
		country = re.match(r'reddit\.([a-zA-Z]*)\.txt\.tok\.clean', country_path).group(1)
		for user in os.listdir(country_full_path):
			paths = []
			full_user_path = os.path.join(country_full_path, user)
			for i, chuck in enumerate(os.listdir(full_user_path)):
				full_chuck_path = os.path.join(full_user_path, chuck)
				paths.append(full_chuck_path)
			users_paths[user] = paths
		countries_paths[country] = users_paths
	return countries_paths


def get_trained_model(features, target, num_threads):
	model = linear_model.LogisticRegression(solver='lbfgs', n_jobs=num_threads, multi_class='auto')
	model.fit(features, target)
	logger.info("Model trained")
	return model


def extract_features_bow(paths, vocab=None, features_count=1000):
	logger.info('Extracting bow')
	vectorizer = TfidfVectorizer(input='filename', encoding='utf-8', decode_error='ignore', max_features=features_count, dtype=np.float32, vocabulary=vocab)
	if vocab is not None:
		logger.info('Using vocab')
		return vectorizer.fit_transform(paths)
	else:
		logger.info('extracting ' + str(features_count) + ' features')
		features = vectorizer.fit_transform(paths)
		return features, vectorizer.vocabulary_


def extract_features_pos(paths, vocab=None, features_count=1000):
	logger.info('Extracting pos')
	vectorizer = TfidfVectorizer(input='filename', encoding='utf-8', decode_error='ignore', ngram_range=(3, 3), max_features=features_count, dtype=np.float32, vocabulary=vocab)
	if vocab is not None:
		logger.info('Using vocab')
		return vectorizer.fit_transform(paths)
	else:
		logger.info('extracting ' + str(features_count) + ' features')
		features = vectorizer.fit_transform(paths)
		return features, vectorizer.vocabulary_


def extract_features_char3(paths, vocab=None, features_count=1000):
	logger.info('Extracting char3')
	vectorizer = TfidfVectorizer(input='filename', encoding='utf-8', decode_error='ignore', ngram_range=(3, 3), analyzer='char', max_features=features_count, dtype=np.float32, vocabulary=vocab)
	if vocab is not None:
		logger.info('Using vocab')
		return vectorizer.fit_transform(paths)
	else:
		logger.info('extracting ' + str(features_count) + ' features')
		features = vectorizer.fit_transform(paths)
		return features, vectorizer.vocabulary_


def write_out_features(country, target_is_native, target_lang_family, target_native_lang, features):
	logger.info('Writing country out of sample features')
	sparse.save_npz(country + "_features.npz", features)
	np.savez_compressed(country + "_targets.npz", is_native=target_is_native, lang_family=target_lang_family, native_lang=target_native_lang)


def read_country_from_file(write_out, country):
	logger.info("Reading country out of sample features from file")
	prefix = os.path.join(write_out, country)
	features = sparse.load_npz(prefix + "_features.npz")
	in_targets = np.load(prefix + "_targets.npz")
	in_is_native_target = in_targets['is_native']
	in_lang_family_target = in_targets['lang_family']
	in_native_lang_target = in_targets['native_lang']
	return features, in_native_lang_target, in_lang_family_target, in_is_native_target


class NLI:
	def __init__(self, text, pos, threads, types):
		self.threads = threads
		self.feature_types = types

		self.model_is_native = None
		self.model_lang_family = None
		self.model_native_lang = None
		self.in_sample_feature = None

		self.in_is_native_target = None
		self.in_lang_family_target = None
		self.in_native_lang_target = None

		if text:
			self.euro_path = os.path.join(text, 'europe_data')
			self.non_euro_path = os.path.join(text, 'non_europe_data')
			logger.info('extracting text in sample paths')
			self.text_chunks_paths = get_chunks_folders(self.euro_path)
			logger.info('extracting text out sample paths')
			self.out_text_chunks_paths = get_chunks_folders(self.non_euro_path)

		if pos:
			self.euro_pos_path = os.path.join(pos, 'europe_data')
			self.non_euro_pos_path = os.path.join(pos, 'non_europe_data')
			logger.info('extracting pos in sample paths')
			self.pos_chunks_paths = get_chunks_folders(self.euro_pos_path)
			logger.info('extracting pos out sample paths')
			self.out_pos_chunks_paths = get_chunks_folders(self.non_euro_pos_path)

		self.vocabs = {}

	def write_in_features(self, file):
		logger.info('Writing in sample features')
		sparse.save_npz(file + "_features.npz", self.in_sample_feature)
		np.savez_compressed(file + "_targets.npz", is_native=self.in_is_native_target, lang_family=self.in_lang_family_target, native_lang=self.in_native_lang_target)

	def load_in_features(self, file):
		logger.info('Loading in sample features')
		self.in_sample_feature = sparse.load_npz(file + "_features.npz")
		in_targets = np.load(file + "_targets.npz")
		self.in_is_native_target = in_targets['is_native']
		self.in_lang_family_target = in_targets['lang_family']
		self.in_native_lang_target = in_targets['native_lang']

	def save_features(self, features, prefix=""):
		for i, mat in enumerate(features):
			logger.info("Saving " + self.feature_types[i] + " features")
			filename = prefix + self.feature_types[i] + "_features.npz"
			sparse.save_npz(filename, mat)

	def set_function_words(self, words):
		logger.info('Setting Function words')
		self.vocabs['fw'] = ({word: i for i, word in enumerate(words)})

	def write_models(self):
		logger.info('Writing models')
		prefix = '-'.join(self.feature_types)
		f1 = open(prefix+"_is_native.pkl", 'wb')
		f2 = open(prefix+"_lang_family.pkl", 'wb')
		f3 = open(prefix+"_native_lang.pkl", 'wb')
		pickle.dump(self.model_is_native, f1)
		pickle.dump(self.model_lang_family, f2)
		pickle.dump(self.model_native_lang, f3)
		f1.close()
		f2.close()
		f3.close()

	def load_models(self):
		logger.info('Loading models')
		prefix = '-'.join(self.feature_types)
		f1 = open(prefix+"_is_native.pkl", 'rb')
		f2 = open(prefix+"_lang_family.pkl", 'rb')
		f3 = open(prefix+"_native_lang.pkl", 'rb')
		self.model_is_native = pickle.load(f1)
		self.model_lang_family = pickle.load(f2)
		self.model_native_lang = pickle.load(f3)
		f1.close()
		f2.close()
		f3.close()

	def dump_vocabs(self):
		logger.info('write vocabs')
		for feature_type in self.feature_types:
			logger.info('writing ' + feature_type + '_vocab.pkl')
			filename = feature_type + '_vocab.pkl'
			f = open(filename, 'wb')
			pickle.dump(self.vocabs[feature_type], f)
			f.close()

	def read_vocabs(self):
		for feature_type in self.feature_types:
			filename = feature_type + '_vocab.pkl'
			logger.info('reading ' + filename)
			f = open(filename, 'rb')
			self.vocabs[feature_type] = (pickle.load(f))
			f.close()

	def features_in_sample(self):
		logger.info('Extracting in sample features')
		all_text_paths = []
		all_pos_paths = []
		for country in sorted(self.text_chunks_paths.keys()):
			all_text_paths.extend(p for s in sorted(self.text_chunks_paths[country].keys()) for p in self.text_chunks_paths[country][s])
		for country in sorted(self.pos_chunks_paths.keys()):
			all_pos_paths.extend(p for s in sorted(self.pos_chunks_paths[country].keys()) for p in self.pos_chunks_paths[country][s])
		logger.info("in sample files: " + str(len(all_text_paths)))

		features_per_type = []
		for feature_type in self.feature_types:
			logger.info('Extracting type:' + feature_type)
			features = None
			vocab = None
			try:
				if feature_type == 'bow':
					features, vocab = extract_features_bow(all_text_paths)
				elif feature_type == 'char3':
					features, vocab = extract_features_char3(all_text_paths)
				elif feature_type == 'pos':
					features, vocab = extract_features_pos(all_pos_paths)
				elif feature_type == 'fw':
					logger.info('extracting fw using bow')
					features = extract_features_bow(all_text_paths, vocab=self.vocabs['fw'])
				else:
					logger.warning('Unknown features type')
			except MemoryError as me:
				logger.error("got memory error")
				logger.exception(me)
				logger.info("saving existing in features")
				self.save_features(features_per_type, prefix="in_")
				raise me

			if vocab is not None:
				self.vocabs[feature_type] = vocab
			features_per_type.append(features)

		self.in_sample_feature = sparse.hstack(features_per_type).tocsr()

	def test_out_paths(self, load_out, write_out):
		logger.info('Extracting out sample features')
		for country in sorted(self.out_text_chunks_paths.keys()):
			country_features = None
			target_native_lang, target_lang_family, target_is_native = None, None, None

			if load_out is None:
				logger.info("Extracting out features from files")
				country_text_paths = [p for s in sorted(self.out_text_chunks_paths[country].keys()) for p in self.out_text_chunks_paths[country][s]]
				country_pos_paths = [p for s in sorted(self.out_pos_chunks_paths[country].keys()) for p in self.out_pos_chunks_paths[country][s]]
				logger.info("out sample files for country: " + country + " " + str(len(country_text_paths)))
				logger.info("Extracting features for country " + country)
				features_per_type = []

				for feature_type in self.feature_types:
					logger.info('Extracting type: ' + feature_type)
					features = None

					if feature_type == 'bow':
						features = extract_features_bow(country_text_paths, vocab=self.vocabs[feature_type])
					elif feature_type == 'char3':
						features = extract_features_char3(country_text_paths, vocab=self.vocabs[feature_type])
					elif feature_type == 'pos':
						features = extract_features_pos(country_pos_paths, vocab=self.vocabs[feature_type])
					elif feature_type == 'fw':
						logger.info('extracting fw using bow')
						features = extract_features_bow(country_text_paths, vocab=self.vocabs['fw'])   # i should always be zero
					else:
						logger.warning('Unknown features type')

					features_per_type.append(features)

				country_features = sparse.hstack(features_per_type)
				target_native_lang, target_lang_family, target_is_native = self.get_target_for_country(country)
			else:
				logger.info("reading out features from file: " + country)
				country_features, target_native_lang, target_lang_family, target_is_native = read_country_from_file(load_out, country)

			if write_out is not None:
				write_out_features(country, target_is_native, target_lang_family, target_native_lang, country_features)

			logger.info("testing results of country " + country)
			logger.info("Native language speaker score:")
			predictions = self.model_native_lang.predict(country_features)
			score = accuracy_score(target_native_lang, predictions)
			logger.info(score)

			logger.info("Language family score:")
			predictions = self.model_lang_family.predict(country_features)
			score = accuracy_score(target_lang_family, predictions)
			logger.info(score)

			logger.info("Is native speaker:")
			predictions = self.model_is_native.predict(country_features)
			score = accuracy_score(target_is_native, predictions)
			logger.info(score)

			gc.collect()

	def target_in_sample(self):
		logger.info('Creating in sample targets')
		temp_in_native_lang = []
		temp_in_lang_family = []
		temp_in_is_native = []
		for country in sorted(self.text_chunks_paths.keys()):
			c_lang = countries_native_family.lang_enum[countries_native_family.country_language[country]]
			lang_family = countries_native_family.family_enum[
				countries_native_family.language_family[countries_native_family.country_language[country]]]
			is_native = int(lang_family == 0)
			for user in sorted(self.text_chunks_paths[country].keys()):
				for _ in self.text_chunks_paths[country][user]:
					temp_in_native_lang.append(c_lang)
					temp_in_lang_family.append(lang_family)
					temp_in_is_native.append(is_native)

		self.in_native_lang_target = np.array(temp_in_native_lang, order='C')
		self.in_lang_family_target = np.array(temp_in_lang_family, order='C')
		self.in_is_native_target = np.array(temp_in_is_native, order='C')

	def get_target_for_country(self, country):
		logger.info('Creating out sample targets')
		temp_out_native_lang = []
		temp_out_lang_family = []
		temp_out_is_native = []
		c_lang = countries_native_family.lang_enum[countries_native_family.country_language[country]]
		lang_family = countries_native_family.family_enum[countries_native_family.language_family[countries_native_family.country_language[country]]]
		is_native = int(lang_family == 0)
		for user in sorted(self.out_text_chunks_paths[country].keys()):
			for _ in self.out_text_chunks_paths[country][user]:
				temp_out_native_lang.append(c_lang)
				temp_out_lang_family.append(lang_family)
				temp_out_is_native.append(is_native)

		return np.array(temp_out_native_lang, order='C'), np.array(temp_out_lang_family, order='C'), np.array(temp_out_is_native, order='C')

	def train(self):
		logger.info('Training model for "Is native speaker"')
		self.model_is_native = get_trained_model(self.in_sample_feature, self.in_is_native_target, self.threads)
		logger.info('Training model for "Native language family"')
		self.model_lang_family = get_trained_model(self.in_sample_feature, self.in_lang_family_target, self.threads)
		logger.info('Training model for "Native Language"')
		self.model_native_lang = get_trained_model(self.in_sample_feature, self.in_native_lang_target, self.threads)

	def calc_10_fold_score(self):
		logger.info('Calculating 10 fold scores')
		score = np.average(cross_val_score(self.model_native_lang, self.in_sample_feature, self.in_native_lang_target, cv=10)) * 100
		logger.info("Native language speaker score: " + str(score))

		score = np.average(cross_val_score(self.model_lang_family, self.in_sample_feature, self.in_lang_family_target, cv=10)) * 100
		logger.info("Language family score: " + str(score))

		score = np.average(cross_val_score(self.model_is_native, self.in_sample_feature, self.in_is_native_target, cv=10)) * 100
		logger.info("Is native speaker: " + str(score))


def main(text_source, pos_source, num_threads, load_in, load_out, write_in, write_out, read_model, feature_types):
	set_log()
	logger.info('start')

	obj = NLI(text_source, pos_source, num_threads, feature_types)

	logger.info('load in from file is: ' + str(load_in))
	if load_in is not None:
		obj.load_in_features(load_in)
		obj.read_vocabs()
	else:
		obj.set_function_words(en_function_words.FUNCTION_WORDS)
		obj.features_in_sample()
		obj.dump_vocabs()
		obj.target_in_sample()

	logger.info('write in to file is: ' + str(write_in))
	if write_in is not None:
		obj.write_in_features(write_in)

	if read_model:
		obj.load_models()
	else:
		obj.train()
		obj.write_models()

	obj.calc_10_fold_score()

	logger.info("Testing out of sample countries")
	obj.test_out_paths(load_out, write_out)


if __name__ == '__main__':
	args = parse_args()
	main(args.text, args.pos, args.threads, args.load_in, args.load_out, args.write_in, args.write_out, args.read_models, args.features)
