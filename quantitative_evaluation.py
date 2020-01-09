import pickle, re
from pprint import pprint
from evaluation import calc_average
from utils import *
import pandas as pd
from collections import Counter
from nltk import tokenize
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction
from scipy.stats import pearsonr, spearmanr
from itertools import combinations
import matplotlib.pyplot as plt

hyp_pattern = re.compile(r'^\<BOS\> (.*?)\<EOS\>$')

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

class Distinct:
    def __init__(self, sentences):
        self.sentences = sentences

    def score(self, n):
        grams = [' '.join(gram) for sentence in self.sentences for gram in self._n_gram(sentence, n)]
        return len(set(grams))/len(grams)

    def _n_gram(self, seq, n):
        return [seq[i:i+n] for i in range(len(seq)-n+1)]

def get_dataframe(result):
    df = pd.DataFrame({'DA_pred':result['DA_preds'],
                       'DA_true':result['DA_trues'],
                       'hyp':[len(hyp.split(' ')) - 2 for hyp in result['hyps']],
                       'ref':[len(ref.split(' ')) - 2 for ref in result['refs']],
                       'hyp_txt':[re.sub(r'\<(.+?)\>\s|\s\<(.+?)\>', '' , sentence) for sentence in result['hyps']],
                       'ref_txt':[re.sub(r'\<(.+?)\>\s|\s\<(.+?)\>', '' , sentence) for sentence in result['refs']]})
    return df

def plotfig(X1, Y1, Y2, xlabel, ylabel, imgname):
    plt.figure(figsize=(40, 30))
    plt.rcParams['font.size'] = 48
    plt.scatter(X1, Y1, c='blue', s=100, label='Merge')
    plt.scatter(X1, Y2, c='red', s=100, label='Separate')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.savefig(os.path.join('./data/images/', imgname))

    
def quantitative_evaluation():
    with open('./data/images/results_hred.pkl', 'rb') as f:
        result = pickle.load(f)
    df_hred = get_dataframe(result)
    with open('./data/images/results_proposal2.pkl', 'rb') as f:
        result = pickle.load(f)
    df_proposal = get_dataframe(result)
    with open('./data/images/results_proposal1.pkl', 'rb') as f:
        result = pickle.load(f)
    df_proposal1 = get_dataframe(result)
    with open('./data/model/utterance_vocab.dict', 'rb') as f:
        vocab = pickle.load(f)
    
    hyp_words_proposal = {tag : Counter([word for sentence in df_proposal[df_proposal['DA_pred'] == tag]['hyp_txt'] for word in sentence.split(' ')]) for tag in set(df_proposal['DA_pred'])}
    ref_words = {tag : Counter([word for sentence in df_proposal[df_proposal['DA_true'] == tag]['ref_txt'] for word in sentence.split(' ')]) for tag in set(df_proposal['DA_true'])}

    hyp_words_hred = {tag : Counter([word for sentence in df_hred[df_hred['DA_true'] == tag]['hyp_txt'] for word in sentence.split(' ')]) for tag in set(df_hred['DA_true'])}
    hyp_words_proposal1 = {tag : Counter([word for sentence in df_proposal1[df_proposal1['DA_pred'] == tag]['hyp_txt'] for word in sentence.split(' ')]) for tag in set(df_proposal1['DA_pred'])}

    # make documents for each dialogue-act
    documents = {tag : ' '.join([sentence for sentence in df_hred[df_hred['DA_true'] == tag]['ref_txt']]) for tag in set(df_hred['DA_true'])}
    # tf_idf = tfidf(document=[sentence for sentence in documents.values()], calc=True)
    tf_idf = tfidf(document=[document for document in documents.values()])
    keywords = {tag: [kwd for kwd in kwds] for kwds, tag in zip(tf_idf.get_topk(50), documents.keys())}
    df_corr = pd.DataFrame({
        'HRED/pearson':[], 'HRED/spearman':[],
        'Merge/pearson':[], 'Merge/spearman':[],
        'Separate/pearson':[], 'Separate/spearman':[]})
    for t in set(df_proposal['DA_true']):
        if t in hyp_words_proposal.keys():
            vocab = keywords[t]
            s_ref = [0 for _ in range(len(vocab))]
            s_proposal = [0 for _ in range(len(vocab))]
            s_hred = [0 for _ in range(len(vocab))]
            s_proposal1 = [0 for _ in range(len(vocab))]
            for w, c in ref_words[t].items():
                if w in vocab:
                    s_ref[vocab.index(w)] = c
            for w, c in hyp_words_proposal[t].items():
                if w in vocab:
                    s_proposal[vocab.index(w)] = c
            for w, c in hyp_words_hred[t].items():
                if w in vocab:
                    s_hred[vocab.index(w)] = c
            for w, c in hyp_words_proposal1[t].items():
                if w in vocab:
                    s_proposal1[vocab.index(w)] = c
            # plotfig(X1=s_ref, Y1=s_proposal, Y2=s_proposal1, xlabel='reference', ylabel='Models', imgname='scatter_{}.png'.format(t))
            df_corr.loc[t] = [pearsonr(s_hred, s_ref)[0], spearmanr(s_hred, s_ref)[0],
                              pearsonr(s_proposal, s_ref)[0], spearmanr(s_proposal, s_ref)[0],
                              pearsonr(s_proposal1, s_ref)[0], spearmanr(s_proposal1, s_ref)[0]]

        else:
            print('No {} in hypothesis'.format(t))

    print(df_corr)
    df_corr.to_csv('./data/images/keyword_correlation.csv')

    df_tag = pd.DataFrame({'pearson':[], 'spearman':[]})
    for a, b in combinations(set(df_proposal['DA_true']), 2):
        s_a = [0 for _ in range(len(vocab))]
        s_b = [0 for _ in range(len(vocab))]
        for w, c in ref_words[a].items():
            s_a[vocab[w]] = c
        for w, c in ref_words[b].items():
            s_b[vocab[w]] = c
        df_tag.loc['{} & {}'.format(a, b)] = [pearsonr(s_a, s_b)[0], spearmanr(s_a, s_b)[0]]

    print(df_tag)
    df_tag.to_csv('./data/images/tagcorrelations.csv')

    mpmi = MPMI(documents=documents)

    df_mpmi = pd.DataFrame({
        'Merge/MPMI':[], 'Separate/MPMI':[]})

    for t in set(df_proposal['DA_true']):
        if t in hyp_words_proposal.keys():
            merge_sentences = [[word for word in sentence.split(' ')] for sentence in df_proposal1[df_proposal1['DA_pred'] == t]['hyp_txt']]
            separate_sentences = [[word for word in sentence.split(' ')] for sentence in df_proposal[df_proposal['DA_pred'] == t]['hyp_txt']]
            df_mpmi.loc[t] = [mpmi.get_score(merge_sentences, t), mpmi.get_score(separate_sentences, t)]
        else:
            print('No {} in hypothesis'.format(t))
    
    print(df_mpmi)
    df_mpmi.to_csv('./data/images/mpmi.csv')

def evaluation(experiment):
    for line in open('./data/result/result_{}.tsv'.format(experiment)).readlines():
        assert len(line.split('\t')) == 3, line
    rl_hyps = [line.strip().split('\t')[1].split(' ') for line in open('./data/result/result_{}.tsv'.format(experiment), 'r').readlines()]
    refs = [[line.strip().split('\t')[2].split(' ')] for line in open('./data/result/result_{}.tsv'.format(experiment), 'r').readlines()]
    for n in range(1, 5):
        print('BLEU-{}: {}'.format(n, corpus_bleu(refs, rl_hyps, weights=[1/n for _ in range(1, n+1)], smoothing_function=SmoothingFunction().method2)))
    rl_dist = Distinct(sentences=[line for line in rl_hyps])
    for n in range(1, 3):
        # print('seq2seq Distinct-{}: {}'.format(n, seq2seq_dist.score(n)))
        print('RL_s2s Distinct-{}: {}'.format(n, rl_dist.score(n)))


if __name__ == '__main__':
    args = parse()
    evaluation(args.expr)

