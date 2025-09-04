# models.py
import nltk
import string
import numpy as np
from nltk.corpus import stopwords
nltk.download("stopwords")
from sentiment_data import *
from utils import *
import random
random.seed(127)

from collections import Counter

class FeatureExtractor(object):
    """
    Feature extraction base type. Takes a sentence and returns an indexed list of features.
    """
    def get_indexer(self):
        raise Exception("Don't call me, call my subclasses")

    def extract_features(self, sentence: List[str], add_to_indexer: bool=False) -> Counter:
        """
        Extract features from a sentence represented as a list of words. Includes a flag add_to_indexer to
        :param sentence: words in the example to featurize
        :param add_to_indexer: True if we should grow the dimensionality of the featurizer if new features are encountered.
        At test time, any unseen features should be discarded, but at train time, we probably want to keep growing it.
        :return: A feature vector. We suggest using a Counter[int], which can encode a sparse feature vector (only
        a few indices have nonzero value) in essentially the same way as a map. However, you can use whatever data
        structure you prefer, since this does not interact with the framework code.
        """
        raise Exception("Don't call me, call my subclasses")

    def remove_rare_words(self, min_count = 2):
        word_counts = Counter()
        corpus = read_blind_sst_examples("data/train.txt")
        for sentence in corpus:
            word_counts.update(sentence)
        vocab = {word for word, count in word_counts.items() if count >= min_count}
        return vocab


class UnigramFeatureExtractor(FeatureExtractor):
    """
    Extracts unigram bag-of-words features from a sentence. It's up to you to decide how you want to handle counts
    and any additional preprocessing you want to do.
    """
    def __init__(self, indexer: Indexer):
        self.indexer = indexer
        self.stop_words = set(stopwords.words("english"))
        self.vocab = self.remove_rare_words()

    def get_indexer(self):
        return self.indexer

    def extract_features(self, sentence: List[str], add_to_indexer: bool=True) -> Counter:
        features = Counter()
        # for word in [word.lower() for word in sentence if word.lower() not in self.stop_words and word in self.vocab]:
        for word in [word.lower() for word in sentence if word.lower() not in self.stop_words]:
            word = word.lower()
            if add_to_indexer:
                index = self.indexer.add_and_get_index(word)
            else:
                index = self.indexer.index_of(word)
                if index ==-1:
                    continue
            features[index] += 1
        return features

class BigramFeatureExtractor(FeatureExtractor):
    """
    Bigram feature extractor analogous to the unigram one.
    """
    def __init__(self, indexer: Indexer):
        self.indexer = indexer
        self.stop_words = set(stopwords.words("english"))

    def get_indexer(self):
        return self.indexer

    def extract_features(self, sentence: List[str], add_to_indexer: bool=False) -> Counter:
        features = Counter()
        tokens = [word.lower() for word in sentence if word.lower() not in self.stop_words]
        bigrams = [f"{tokens[i]}|{tokens[i+1]}" for i in range(len(tokens)-1)]
        for bigram in bigrams:
            index = (self.indexer.add_and_get_index(bigram) if add_to_indexer else self.indexer.index_of(bigram))
            if index != -1:
                features[index] += 1
        
        return features


class BetterFeatureExtractor(FeatureExtractor):
    """
    Better feature extractor...try whatever you can think of!
    """
    def __init__(self, indexer: Indexer):
        self.indexer = indexer
        self.stop_words = set(stopwords.words("english"))
        self.negative_words = ("not", "no", "never", "n't")

    def get_indexer(self):
        return self.indexer

    def extract_features(self, sentence: List[str], add_to_indexer: bool=False) -> Counter:
        features = Counter()
        lowered_tokens = [word.lower() for word in sentence]
        contextualized_tokens = self._negate(lowered_tokens)
        bigrams = [f"{lowered_tokens[i]}|{lowered_tokens[i+1]}" for i in range(len(lowered_tokens)-1)]
        for bigram in bigrams:
            index = (self.indexer.add_and_get_index(bigram) if add_to_indexer else self.indexer.index_of(bigram))
            if index != -1:
                features[index] += 1
        
        for word in [word.lower() for word in contextualized_tokens if word.lower() not in self.stop_words]:
            word = word.lower()
            if add_to_indexer:
                index = self.indexer.add_and_get_index(word)
            else:
                index = self.indexer.index_of(word)
                if index ==-1:
                    continue
            features[index] += 1
        return features

    def _negate(self, tokens: List[str]):
        negated_tokens = []
        negate = False
        for token in tokens:
            if token in self.negative_words:
                negate = True
                negated_tokens.append(token)
            elif negate:
                if token in string.punctuation:
                    negate = False
                else:
                    negated_tokens.append(f"NEGATIVE_{token}")
            else:
                negated_tokens.append(token)
        return negated_tokens

class SentimentClassifier(object):
    """
    Sentiment classifier base type
    """
    def predict(self, sentence: List[str]) -> int:
        """
        :param sentence: words (List[str]) in the sentence to classify
        :return: Either 0 for negative class or 1 for positive class
        """
        raise Exception("Don't call me, call my subclasses")


class TrivialSentimentClassifier(SentimentClassifier):
    """
    Sentiment classifier that always predicts the positive class.
    """
    def predict(self, sentence: List[str]) -> int:
        return 1


class PerceptronClassifier(SentimentClassifier):
    """
    Implement this class -- you should at least have init() and implement the predict method from the SentimentClassifier
    superclass. Hint: you'll probably need this class to wrap both the weight vector and featurizer -- feel free to
    modify the constructor to pass these in.
    """
    def __init__(self, weights: Counter, featurizer: FeatureExtractor):
        self.weights = weights
        self.featurizer = featurizer

    def predict(self, sentence: List[str]) -> int:
        features = self.featurizer.extract_features(sentence, add_to_indexer=False)
        value = sum(self.weights[f] * v for f,v in features.items())
        return 1 if value > 0 else 0

class LogisticRegressionClassifier(SentimentClassifier):
    """
    Implement this class -- you should at least have init() and implement the predict method from the SentimentClassifier
    superclass. Hint: you'll probably need this class to wrap both the weight vector and featurizer -- feel free to
    modify the constructor to pass these in.
    """
    def __init__(self, weights: Counter, featurizer: FeatureExtractor):
        self.weights = weights
        self.featurizer = featurizer

    def predict(self, sentence: List[str]) -> int:
        features = self.featurizer.extract_features(sentence, add_to_indexer=False)
        value = sum(self.weights[f] * v for f,v in features.items())
        probability = 1/(1 + np.exp(-value))
        return 1 if probability > 0.5 else 0


def train_perceptron(train_exs: List[SentimentExample], feat_extractor: FeatureExtractor, num_epochs:int = 150, init_alpha:float = 1.0) -> PerceptronClassifier:
    """
    Train a classifier with the perceptron.
    :param train_exs: training set, List of SentimentExample objects
    :param feat_extractor: feature extractor to use
    :return: trained PerceptronClassifier model
    """
    # may want to suffle train_exs for each epoch
    weights = Counter()
    alpha = init_alpha
    for epoch in range(num_epochs):
        shuffled_examples = random.sample(train_exs, len(train_exs))
        for example in shuffled_examples:
            features = feat_extractor.extract_features(example.words, True)
            value = sum(weights[f] * v for f,v in features.items())
            y_pred = 1 if value > 0 else 0

            if y_pred != example.label:
                for f,v in features.items():
                    shift = alpha*v
                    if example.label == 1:
                        weights[f] += shift
                    else:
                        weights[f] -= shift
    
    return PerceptronClassifier(weights, feat_extractor)


def train_logistic_regression(train_exs: List[SentimentExample], feat_extractor: FeatureExtractor, num_epochs:int = 180, init_alpha:float = 85e-4) -> LogisticRegressionClassifier:
    """
    Train a logistic regression model.
    :param train_exs: training set, List of SentimentExample objects
    :param feat_extractor: feature extractor to use
    :return: trained LogisticRegressionClassifier model
    """
    # may want to suffle train_exs for each epoch
    alpha = init_alpha
    weights = Counter()
    for epoch in range(num_epochs):
        shuffled_examples = random.sample(train_exs, len(train_exs))
        for example in shuffled_examples:
            features = feat_extractor.extract_features(example.words, True)
            value = sum(weights[f] * v for f,v in features.items())
            probability = 1 / (1 + np.exp(-value))
            error = example.label - probability

            for f, v in features.items():
                weights[f] += alpha*error*v

        # if epoch > 0:
        #     alpha = alpha - (alpha/num_epochs)
        # print(f"{alpha:.10f}")
    
    return LogisticRegressionClassifier(weights, feat_extractor)


def train_model(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample]) -> SentimentClassifier:
    """
    Main entry point for your modifications. Trains and returns one of several models depending on the args
    passed in from the main method. You may modify this function, but probably will not need to.
    :param args: args bundle from sentiment_classifier.py
    :param train_exs: training set, List of SentimentExample objects
    :param dev_exs: dev set, List of SentimentExample objects. You can use this for validation throughout the training
    process, but you should *not* directly train on this data.
    :return: trained SentimentClassifier model, of whichever type is specified
    """
    # Initialize feature extractor
    if args.model == "TRIVIAL":
        feat_extractor = None
    elif args.feats == "UNIGRAM":
        feat_extractor = UnigramFeatureExtractor(Indexer())
    elif args.feats == "BIGRAM":
        # Add additional preprocessing code here
        feat_extractor = BigramFeatureExtractor(Indexer())
    elif args.feats == "BETTER":
        # Add additional preprocessing code here
        feat_extractor = BetterFeatureExtractor(Indexer())
    else:
        raise Exception("Pass in UNIGRAM, BIGRAM, or BETTER to run the appropriate system")

    # Train the model
    if args.model == "TRIVIAL":
        model = TrivialSentimentClassifier()
    elif args.model == "PERCEPTRON":
        model = train_perceptron(train_exs, feat_extractor, 150, 1)
    elif args.model == "LR":
        model = train_logistic_regression(train_exs, feat_extractor, 100, .0085)
    else:
        raise Exception("Pass in TRIVIAL, PERCEPTRON, or LR to run the appropriate system")
    return model