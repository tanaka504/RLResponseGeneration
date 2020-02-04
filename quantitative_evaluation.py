from utils import *
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
import random, json
import numpy as np

random.seed(42)

def evaluation(experiment):
    jsondata = json.load(open('./data/result/result_{}.json'.format(experiment)))
    rl_hyps, refs = zip(*[(line['hyp'].split(' '), [line['ref'].split(' ')]) for line in jsondata])
    q_hyps, q_refs = zip(*[(line['hyp'].split(' '), [line['ref'].split(' ')]) for line in jsondata if line['da_context'][-1] in ['question', '<Question>']])
    # rl_hyps = [line.strip().split('\t')[1].split(' ') for line in open('./data/result/result_{}.tsv'.format(experiment), 'r').readlines()]
    # refs = [[line.strip().split('\t')[2].split(' ')] for line in open('./data/result/result_{}.tsv'.format(experiment), 'r').readlines()]
    print('avg. length: {}'.format(np.mean([len(line) for line in rl_hyps])))
    print('avg. length of reference: {}'.format(np.mean([len(line[0]) for line in refs])))
    for n in range(1, 5):
        print('BLEU-{}: {}'.format(n, corpus_bleu(refs, rl_hyps, weights=[1/n for _ in range(1, n+1)], smoothing_function=SmoothingFunction().method2)))
    print('Q-BLEU: {}'.format(corpus_bleu(q_refs, q_hyps, smoothing_function=SmoothingFunction().method2)))
    rl_dist = Distinct(sentences=[line for line in rl_hyps])
    ref_dist = Distinct(sentences=[line[0] for line in refs])
    for n in range(1, 3):
        print('RL_s2s Distinct-{}: {}'.format(n, rl_dist.score(n)))
        print('reference distinct-{}: {}'.format(n, ref_dist.score(n)))


def make_human_evaluation():
    proposal_hyps = [line.strip().split('\t')[1] for line in open('./data/result/result_RL_dd.tsv').readlines()]
    ssn_hyps = [line.strip().split('\t')[1] for line in open('./data/result/result_RL_dd_ssn.tsv').readlines()]
    raw_hyps = [line.strip().split('\t')[1] for line in open('./data/result/result_HRED_dd.tsv').readlines()]
    contexts = [line.strip().split('\t')[0].split('|')[-3:] for line in open('./data/result/result_RL_dd.tsv').readlines()]

    candidates = []
    for (raw, ssn, proposal, context) in zip(raw_hyps, ssn_hyps, proposal_hyps, contexts):
        if '<UNK>' in raw or '<UNK>' in ssn or '<UNK>' in proposal or '<UNK>' in ' '.join(context):
            continue
        if raw == proposal or ssn == proposal:
            continue
        candidates.append((context, raw, ssn, proposal))

    random.shuffle(candidates)
    out_f = open('./data/human_evaluation/annotation_data.tsv', 'w')
    test_f = open('./data/human_evaluation/annotation_test.tsv', 'w')
    log_f = open('./data/human_evaluation/log_data.tsv','w')
    log_test_f = open('./data/human_evaluation/log_test.tsv', 'w')
    for candidate in candidates[:100]:
        context, raw, ssn, proposal = candidate
        responses = {
                'raw': raw,
                'ssn': ssn,
                'proposal': proposal,
                }
        indexes = ['raw', 'ssn', 'proposal']
        random.shuffle(indexes)
        log_f.write('\t'.join(indexes) + '\n')
        shuffled = [responses[method] for method in indexes]
        out_f.write('\t'.join(context) + '\t' + '\t'.join(shuffled) + '\n')
    for candidate in candidates[-20:]:
        context, raw, ssn, proposal = candidate
        responses = {
                'raw': raw,
                'ssn': ssn,
                'proposal': proposal,
                }
        indexes = ['raw', 'ssn', 'proposal']
        random.shuffle(indexes)
        log_test_f.write('\t'.join(indexes) + '\n')
        shuffled = [responses[method] for method in indexes]
        test_f.write('\t'.join(context) + '\t' + '\t'.join(shuffled) + '\n')


def human_evaluation():
    logs = [line.strip().split('\t') for line in open('./data/human_evaluation/log_test.tsv').readlines()]
    annotator1 = [line.strip().split('\t')[-6:] for i, line in enumerate(open('./data/human_evaluation/annotation_test20.txt').readlines()) if i > 1]
    naturalness = {'raw': [], 'ssn': [], 'proposal': []}
    relevance = {'raw': [], 'ssn': [], 'proposal': []}
    for log, annotate in zip(logs, annotator1):
        methods = {'A': log[0], 'B': log[1], 'C': log[2]}

        for i, letter in enumerate(annotate[:3]):
            naturalness[methods[letter]].append(2-i)

        for i, letter in enumerate(annotate[3:]):
            relevance[methods[letter]].append(2-i)
    for method in ['raw', 'ssn', 'proposal']:
        print('Naturalness: ', np.mean(naturalness[method]))
        print('Relevance: ', np.mean(relevance[method]))

if __name__ == '__main__':
    args = parse()
    evaluation(args.expr)

