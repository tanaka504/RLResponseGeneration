from utils import *
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction


def evaluation(experiment):
    for line in open('./data/result/result_{}.tsv'.format(experiment)).readlines():
        assert len(line.split('\t')) == 3, line
    rl_hyps = [line.strip().split('\t')[1].split(' ') for line in open('./data/result/result_{}.tsv'.format(experiment), 'r').readlines()]
    refs = [[line.strip().split('\t')[2].split(' ')] for line in open('./data/result/result_{}.tsv'.format(experiment), 'r').readlines()]
    for n in range(1, 5):
        print('BLEU-{}: {}'.format(n, corpus_bleu(refs, rl_hyps, weights=[1/n for _ in range(1, n+1)], smoothing_function=SmoothingFunction().method2)))
    rl_dist = Distinct(sentences=[line for line in rl_hyps])
    for n in range(1, 3):
        print('RL_s2s Distinct-{}: {}'.format(n, rl_dist.score(n)))



if __name__ == '__main__':
    args = parse()
    evaluation(args.expr)

