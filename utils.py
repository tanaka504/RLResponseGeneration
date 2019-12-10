import os, re, json, math
import matplotlib.pyplot as plt
import torch
from nltk import tokenize
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
import pickle
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
from gensim import corpora
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
import argparse
import pyhocon

EOS_token = '<EOS>'
BOS_token = '<BOS>'
parallel_pattern = re.compile(r'^(.+?)(\t)(.+?)$')
file_pattern = re.compile(r'^sw\_([0-9]+?)\_([0-9]+?)\.jsonlines$')

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--expr', '-e', default='seq2seq', help='input experiment config')
    parser.add_argument('--gpu', '-g', type=int, default=0, help='input gpu num')
    parser.add_argument('--epoch', default='trainbest')
    args = parser.parse_args()
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
    return args


def initialize_env(name):
    corpus_path = {
        'jaist': {'path': './data/corpus/jaist', 'pattern': r'^data([0-9]*?)\_{}\_([0-9]*?)\.jsonlines$'},
        'swda': {'path': './data/corpus/json_data', 'pattern': r'^sw_{}_([0-9]*?)\.jsonlines$'},
        'opensubtitles': {'path': './data/corpus/OpenSubtitles', 'pattern': r'^OpenSubtitles\_{}\_([0-9]*?)\.jsonlines$'},
        'dailydialog': {'path': './data/corpus/dailydialog', 'pattern': r'^DailyDialog\_{}\_([0-9]*?)\.jsonlines$'}
    }
    config = pyhocon.ConfigFactory.parse_file('experiments.conf')[name]
    config['log_dir'] = os.path.join(config['log_root'], name)
    config['train_path'] = corpus_path[config['corpus']]['path']
    config['corpus_pattern'] = corpus_path[config['corpus']]['pattern']
    if not os.path.exists(config['log_dir']):
        os.makedirs(config['log_dir'])
    return config

class da_Vocab:
    def __init__(self, config, das=[], create_vocab=True):
        self.word2id = None
        self.id2word = None
        self.config = config
        self.das = das
        if create_vocab:
            self.construct()
        else:
            self.load()

    def construct(self):
        vocab = {'<PAD>': 0, }
        vocab_count = {}

        for token in self.das:
            if token in vocab_count:
                vocab_count[token] += 1
            else:
                vocab_count[token] = 1

        for k, _ in sorted(vocab_count.items(), key=lambda x: -x[1]):
            vocab[k] = len(vocab)
        self.word2id = vocab
        self.id2word = {v : k for k, v in vocab.items()}

        return vocab

    def tokenize(self, X_tensor):
        X_tensor = [[self.word2id[token] for token in sentence] for sentence in X_tensor]
        return X_tensor

    def save(self):
        pickle.dump(self.word2id, open(os.path.join(self.config['log_root'], 'da_vocab.dict'), 'wb'))

    def load(self):
        self.word2id = pickle.load(open(os.path.join(self.config['log_root'], 'da_vocab.dict'), 'rb'))
        self.id2word = {v: k for k, v in self.word2id.items()}

class utt_Vocab:
    def __init__(self, config, sentences=[], create_vocab=True):
        self.word2id = None
        self.id2word = None
        self.config = config
        self.sentences = sentences
        if create_vocab:
            self.construct()
        else:
            self.load()

    def construct(self):
        vocab = {'<UNK>': 0, '<EOS>': 1, '<BOS>': 2, '<PAD>': 3, '<SEP>': 4}
        vocab_count = {}

        for sentence in self.sentences:
            for word in sentence:
                if word in vocab: continue
                if word in vocab_count:
                    vocab_count[word] += 1
                else:
                    vocab_count[word] = 1

        for k, _ in sorted(vocab_count.items(), key=lambda x: -x[1]):
            vocab[k] = len(vocab)
            if len(vocab) >= self.config['UTT_MAX_VOCAB']: break
        self.word2id = vocab
        self.id2word = {v : k for k, v in vocab.items()}
        return vocab

    def tokenize(self, X_tensor):
        X_tensor = [[[self.word2id[token] if token in self.word2id else self.word2id['<UNK>'] for token in seq] for seq in dialogue] for dialogue in X_tensor]
        return X_tensor

    def save(self):
        pickle.dump(self.word2id, open(os.path.join(self.config['log_root'], 'utterance_vocab.dict'), 'wb'))

    def load(self):
        self.word2id = pickle.load(open(os.path.join(self.config['log_root'], 'utterance_vocab.dict'), 'rb'))
        self.id2word = {v: k for k, v in self.word2id.items()}


class tfidf:
    def __init__(self, document):
        self.doc = document
        self.model = TfidfVectorizer(min_df=0.03)
        self.tf_idf = self.model.fit_transform(self.doc).toarray()
        index = self.tf_idf.argsort(axis=1)[:,::-1]
        feature_names = np.array(self.model.get_feature_names())
        self.feature_words = feature_names[index]

    def get_topk(self, k):
        return self.feature_words[:, :k]

class MPMI:
    def __init__(self, documents):
        self.docs = {tag: [word for word in doc.split(' ')] for tag, doc in documents.items()}
        self.vocab = corpora.Dictionary([doc for doc in self.docs.values()])
        self.tag_idx = {tag: idx for idx, tag in enumerate(self.docs.keys())}
        self._count()

    def _count(self):
        N = len([word for doc in self.docs.values() for word in doc])
        counts = {tag : Counter(doc) for tag, doc in self.docs.items()}
        overall_counts = Counter([word for doc in self.docs.values() for word in doc])
        matrix = [[None for _ in self.vocab.token2id] for _ in counts.keys()]
        for tidx, (tag, count) in enumerate(counts.items(), 1):
            for widx, (word, freq) in enumerate(count.items(), 1):
                # print('\rcalculating {}/{} words in {}/{} tags'.format(widx, len(count), tidx, len(counts)), end='')
                Pxy = freq / len(self.docs[tag])
                Px = overall_counts[word] / N
                PMI = math.log(Pxy / Px, 2)
                matrix[self.tag_idx[tag]][self.vocab.token2id[word]] = max(PMI, 0)
        self.matrix = matrix
        print()

    def get_score(self, sentences, tag):
        if len(sentences) < 1:
            return 0
        else:
            return sum(sum(self.matrix[self.tag_idx[tag]][self.vocab.token2id[word]] for word in sentence if word in self.vocab.token2id and not self.matrix[self.tag_idx[tag]][self.vocab.token2id[word]] is None)/ len(sentence) for sentence in sentences) / len(sentences)


def calc_bleu(refs, hyps):
        refs = [[list(map(str, ref))] for ref in refs]
        hyps = [list(map(str, hyp)) for hyp in hyps]
        # try:
        bleu = corpus_bleu(refs, hyps, smoothing_function=SmoothingFunction().method2)
        # except:
        #     bleu = 1e-10
        return bleu


def create_traindata(config, prefix='train'):
    file_pattern = re.compile(config['corpus_pattern'].format(prefix))
    files = [f for f in os.listdir(config['train_path']) if file_pattern.match(f)]
    da_posts = []
    da_cmnts = []
    utt_posts = []
    utt_cmnts = []
    turn = []
    # 1file 1conversation
    for filename in files:
        with open(os.path.join(config['train_path'], filename), 'r') as f:
            data = f.read().split('\n')
            data.remove('')
            da_seq = []
            utt_seq = []
            turn_seq = []
            # 1line 1turn
            for idx, line in enumerate(data, 1):
                jsondata = json.loads(line)
                for da, utt in zip(jsondata['DA'], jsondata['sentence']):
                    if config['lang'] == 'en':
                        utt = [BOS_token] + en_preprocess(utt) + [EOS_token]
                    else:
                        utt = [BOS_token] + utt.split(' ') + [EOS_token]
                    da_seq.append(da)
                    utt_seq.append(utt)
                    turn_seq.append(0)
                turn_seq[-1] = 1
            da_seq = [da for da in da_seq]
        if len(da_seq) <= config['window_size']: continue
        for i in range(max(1, len(da_seq) - 1 - config['window_size'])):
            assert len(da_seq[i:min(len(da_seq)-1, i + config['window_size'])]) >= config['window_size'], filename
            da_posts.append(da_seq[i:min(len(da_seq)-1, i + config['window_size'])])
            da_cmnts.append(da_seq[1 + i:min(len(da_seq), 1 + i + config['window_size'])])
            utt_posts.append(utt_seq[i:min(len(da_seq)-1, i + config['window_size'])])
            utt_cmnts.append(utt_seq[1 + i:min(len(da_seq), 1 + i + config['window_size'])])
            turn.append(turn_seq[i:min(len(da_seq), i + config['window_size'])])
    assert len(da_posts) == len(da_cmnts), 'Unexpect length da_posts and da_cmnts'
    assert len(utt_posts) == len(utt_cmnts), 'Unexpect length utt_posts and utt_cmnts'
    assert all(len(ele) == config['window_size'] for ele in da_posts), {len(ele) for ele in da_posts}
    return da_posts, da_cmnts, utt_posts, utt_cmnts

def en_preprocess(utterance):
    if utterance == '': return ['<Silence>']
    return tokenize.word_tokenize(utterance.lower())

def NLILoader(config, prefix='train'):
    if config['lang'] == 'en':
        tag2id = {'positive': 0, 'neutral': 1, 'negative': 2}
        jsondata = json.load(open('./data/corpus/dnli/dialogue_nli/dialogue_nli_{}.jsonl'.format(prefix)))
        X = []
        Y = []
        for line in jsondata:
            x1 = line['sentence1']
            x2 = line['sentence2']
            label = line['label']
            X.append((x1, x2))
            Y.append(tag2id[label])
    else:
        tag2id = {'I': 0, 'B': 1, 'F': 2, 'C': 3}
        jsondata = json.load(open('./data/corpus/RITE/RITE2_JA_{}_mc.json'.format(prefix)))
        X = []
        Y = []
        for line in jsondata:
            x1 = line['t1']
            x2 = line['t2']
            label = line['label']
            X.append((x1, x2))
            Y.append(tag2id[label])
    return X, Y

def MTLoader():
    X = []
    Y = []
    for idx, line in enumerate(open('./data/corpus/mt.tsv', 'r').readlines()):
        if idx == 20500: break
        print('\r{}'.format(idx), end='')
        x, y = line.strip().split('\t')
        X.append(x.split(' '))
        Y.append(y.split(' '))
    return X, Y
