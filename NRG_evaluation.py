import math
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from collections import Counter
from gensim import corpora


class BLEU_score:
    def __init__(self):
        pass

    def get_bleu_n(self, refs, hyps, n):
        BLEU_prec = np.mean([max([self._calc_bleu(ref, hyp, n) for ref in refs]) for hyp in hyps])
        BLEU_recall = np.mean([max([self._calc_bleu(ref, hyp, n) for hyp in hyps]) for ref in refs])
        return BLEU_prec, BLEU_recall

    def _calc_bleu(self, ref, hyp, n):
        try:
            return sentence_bleu(references=[ref], hypothesis=hyp, smoothing_function=SmoothingFunction().method7, weights=[1/n for _ in range(1, n+1)])
        except:
            return 0.0

class BoW_score:
    def __init__(self, w2v_path, vocab):
        self.w2v_path = w2v_path
        self.vocab = vocab
        self._load_w2v()

    def greedy_matching_score(self, ref, hyp, w2v):
        res1 = self.oneside_greedy_matching_score(ref, hyp, w2v)
        res2 = self.oneside_greedy_matching_score(hyp, ref, w2v)
        res_sum = (res1 + res2) / 2.0
        return res_sum

    def oneside_greedy_matching_score(self, ref, hyp, w2v):
        dim = list(w2v.values())[0].shape[0]  # embedding dimensions
        y_count = 0
        x_count = 0
        o = 0.0
        Y = np.zeros((dim, 1))
        for tok in hyp:
            if tok in w2v:
                Y = np.hstack((Y, (w2v[tok].reshape((dim, 1)))))
                y_count += 1

        for tok in ref:
            if tok in w2v:
                x = w2v[tok]
                tmp = cosine_similarity(Y.T, x.reshape(1, -1))
                o += np.max(tmp)
                x_count += 1

        # if none of the words in response or ground truth have embeddings, count result as zero
        if x_count < 1 or y_count < 1:
            return 0

        o /= float(x_count)
        return o

    def vector_extrema_score(self, ref, hyp, w2v):
        X, Y = [], []
        x_cnt, y_cnt = 0, 0
        for tok in ref:
            if tok in w2v:
                X.append(w2v[tok])
                x_cnt += 1
        for tok in hyp:
            if tok in w2v:
                Y.append(w2v[tok])
                y_cnt += 1

        # if none of the words in ground truth have embeddings, skip
        if x_cnt == 0:
            return 0

        # if none of the words have embeddings in response, count result as zero
        if y_cnt == 0:
            return 0

        xmax = np.max(X, 0)  # get positive max
        xmin = np.min(X, 0)  # get abs of min
        xtrema = []
        for i in range(len(xmax)):
            if np.abs(xmin[i]) > xmax[i]:
                xtrema.append(xmin[i])
            else:
                xtrema.append(xmax[i])
        X = np.array(xtrema)  # get extrema

        ymax = np.max(Y, 0)
        ymin = np.min(Y, 0)
        ytrema = []
        for i in range(len(ymax)):
            if np.abs(ymin[i]) > ymax[i]:
                ytrema.append(ymin[i])
            else:
                ytrema.append(ymax[i])
        Y = np.array(ytrema)

        o = cosine_similarity(Y.reshape(1, -1), X.reshape(1, -1))[0][0]

        return o

    def embedding_average_score(self, ref, hyp, w2v):
        dim = list(w2v.values())[0].shape[0]  # embedding dimensions
        X = np.zeros((dim,))
        x_cnt, y_cnt = 0, 0
        for tok in ref:
            if tok in w2v:
                X += w2v[tok]
                x_cnt += 1
        Y = np.zeros((dim,))
        for tok in hyp:
            if tok in w2v:
                Y += w2v[tok]
                y_cnt += 1

        # if none of the words in ground truth have embeddings, skip
        if x_cnt == 0:
            return 0

        # if none of the words have embeddings in response, count result as zero
        if y_cnt == 0:
            return 0

        X = np.array(X) / x_cnt
        Y = np.array(Y) / y_cnt
        o = cosine_similarity(Y.reshape(1, -1), X.reshape(1, -1))[0][0]
        return o

    def get_score(self, refs, hyps):
        A_bow_prec = np.mean([max([self.embedding_average_score(ref, hyp, self.word2vec) for ref in refs]) for hyp in hyps])
        A_bow_recall = np.mean([max([self.embedding_average_score(ref, hyp, self.word2vec) for hyp in hyps]) for ref in refs])
        E_bow_prec = np.mean([max([self.vector_extrema_score(ref, hyp, self.word2vec) for ref in refs]) for hyp in hyps])
        E_bow_recall = np.mean([max([self.vector_extrema_score(ref, hyp, self.word2vec) for hyp in hyps]) for ref in refs])
        # G_bow_prec = np.mean([max([self.greedy_matching_score(ref, hyp, self.word2vec) for ref in refs]) for hyp in hyps])
        # G_bow_recall = np.mean([max([self.greedy_matching_score(ref, hyp, self.word2vec) for hyp in hyps]) for ref in refs])
        return (A_bow_prec, A_bow_recall), (E_bow_prec, E_bow_recall)

    def _load_w2v(self):
        if self.w2v_path is None:
            return
        with open(self.w2v_path, "rb") as f:
            lines = f.readlines()
        raw_word2vec = {}
        for l in lines:
            w, vec = l.split(" ", 1)
            raw_word2vec[w] = vec
        # clean up lines for memory efficiency
        self.word2vec = {}
        oov_cnt = 0
        for v in self.vocab:
            str_vec = raw_word2vec.get(v, None)
            if str_vec is None:
                oov_cnt += 1
            else:
                vec = np.fromstring(str_vec, sep=" ")
                self.word2vec[v] = vec
        print("word2vec cannot cover %f vocab" % (float(oov_cnt) / len(self.vocab)))

class Distinct:
    def __init__(self, sentences):
        self.sentences = sentences

    def score(self, n):
        grams = [' '.join(gram) for sentence in self.sentences for gram in self._n_gram(sentence, n)]
        return len(set(grams))/len(grams)

    def _n_gram(self, seq, n):
        return [seq[i:i+n] for i in range(len(seq)-n+1)]

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
