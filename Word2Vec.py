#!/usr/bin/env python3

from sklearn.base import BaseEstimator, TransformerMixin
from gensim.models import Word2Vec
import numpy as np


class Word2VecTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, vector_size=100, window=5, min_count=1, workers=4):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.model = None

    def fit(self, X, y=None):
        sentences = [sentence.split() for sentence in X]
        self.model = Word2Vec(
            sentences,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=self.workers,
        )
        return self

    def transform(self, X):
        sentences = [sentence.split() for sentence in X]
        features = np.array([self.sentence_vector(sentence) for sentence in sentences])
        return features

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

    def sentence_vector(self, sentence):
        # Average word vectors for the sentence
        vectors = [self.model.wv[word] for word in sentence if word in self.model.wv]
        if vectors:
            return np.mean(vectors, axis=0)
        else:
            # Return a zero vector if no words are found in the model
            return np.zeros(self.vector_size)

    def get_feature_names_out(self):
        return list(self.model.wv.index_to_key)
