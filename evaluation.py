from models import *
from nn_blocks import *
from utils import *
from train import Reward
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, accuracy_score
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import json


sentence_pattern = re.compile(r'<BOS> (.*?) <EOS>')
def evaluate(experiment):
    print('load vocab')
    config = initialize_env(experiment)
    X_test, Y_test, XU_test, YU_test, turn = create_traindata(config=config, prefix='test')
    # X_test, Y_test, XU_test, YU_test, turn = create_traindata(config=config, prefix='train')
    da_vocab = da_Vocab(config=config, create_vocab=False)
    utt_vocab = utt_Vocab(config=config, create_vocab=False)
    X_test, Y_test = da_vocab.tokenize(X_test), da_vocab.tokenize(Y_test)
    XU_test, YU_test = utt_vocab.tokenize(XU_test), utt_vocab.tokenize(YU_test)

    print('load models')
    reward_fn = Reward(utt_vocab=utt_vocab, da_vocab=da_vocab, config=config)
    model = RL(utt_vocab=utt_vocab,
               da_vocab=da_vocab,
               fine_tuning=False,
               reward_fn=reward_fn,
               criterion=nn.CrossEntropyLoss(ignore_index=utt_vocab.word2id['<PAD>'], reduce=False),
               config=config).cuda()
    model.load_state_dict(torch.load(os.path.join(config['log_dir'], 'statevalidbest.model'), map_location=lambda storage, loc: storage))
    # model.load_state_dict(torch.load(os.path.join(config['log_root'], 'HRED_dd_pretrain', 'statevalidbest.model'), map_location=lambda storage, loc: storage))

    contradict = Contradict(da_vocab=da_vocab, utt_vocab=utt_vocab, config=config)
    c_perplexity = contradict.evaluate(model)
    indexes = [i for i in range(len(XU_test))]
    batch_size = config['BATCH_SIZE']
    results = []
    k = 0
    nli_rwds = []
    ssn_rwds = []
    da_rwds = []
    out_f = open('./data/result/result_{}.tsv'.format(experiment), 'w')
    while k < len(indexes):
        step_size = min(batch_size, len(indexes) - k)
        batch_idx = indexes[k : k + step_size]
        print('\r{}/{} conversation evaluating'.format(k + step_size, len(X_test)), end='')

        X_seq = [X_test[seq_idx] for seq_idx in batch_idx]
        Y_seq = [Y_test[seq_idx] for seq_idx in batch_idx]
        XU_seq = [XU_test[seq_idx] for seq_idx in batch_idx]
        YU_seq = [YU_test[seq_idx] for seq_idx in batch_idx]
        turn_seq = [turn[seq_idx] for seq_idx in batch_idx]
        assert len(X_seq) == len(Y_seq), 'Unexpect sequence len in test data'
        max_conv_len = max(len(s) for s in XU_seq)
        X_tensor = []
        XU_tensor = []
        turn_tensor = []
        for i in range(0, max_conv_len):
            max_xseq_len = max(len(XU[i]) + 1 for XU in XU_seq)
            for ci in range(len(XU_seq)):
                XU_seq[ci][i] = XU_seq[ci][i] + [utt_vocab.word2id['<PAD>']] * (max_xseq_len - len(XU_seq[ci][i]))
            X_tensor.append([[x[i]] for x in X_seq])
            XU_tensor.append(torch.tensor([XU[i] for XU in XU_seq]).cuda())
            turn_tensor.append([[t[i]] for t in turn_seq])
        XU_tensor = [XU_tensor[-1]]
        pred_seq = model.predict(X_utt=XU_tensor, step_size=step_size)
        reward = reward_fn.reward(hyp=pred_seq, ref=None, context=[[s for s in X.data.tolist()] for X in XU_tensor], da_context=X_tensor, turn=turn_tensor, step_size=step_size)
        nli_rwds.append(reward_fn.rewards['nli'])
        ssn_rwds.append(reward_fn.rewards['ssn'])
        da_rwds.append(reward_fn.rewards['da_rwd'])
        Y_tensor = [y[-1] for y in Y_seq]
        YU_tensor = [y[-1] for y in YU_seq]
        for bidx in range(len(XU_seq)):
            hyp = text_postprocess(' '.join([utt_vocab.id2word[wid] for wid in pred_seq[bidx]]))
            ref = text_postprocess(' '.join(utt_vocab.id2word[wid] for wid in YU_tensor[bidx]))
            contexts = [text_postprocess(' '.join([utt_vocab.id2word[wid] for wid in sent])) for sent in XU_seq[bidx]]
            results.append({
                'hyp': hyp,
                'ref': ref,
                'context': contexts,
            })
            out_f.write('{}\t{}\t{}\n'.format('|'.join(contexts), hyp, ref))
        k += step_size
    print()
    nli_rwd = np.mean([score for ele in nli_rwds for score in ele])
    nli_std = np.std([score for ele in nli_rwds for score in ele])
    ssn_rwd = np.mean([score for ele in ssn_rwds for score in ele])
    ssn_std = np.std([score for ele in ssn_rwds for score in ele])
    da_rwd = np.mean([score for ele in da_rwds for score in ele])
    da_std = np.std([score for ele in da_rwds for score in ele])
    print('nli: {}, ssn: {}, da: {}, contradict-perplexity: {}'.format(nli_rwd, ssn_rwd, da_rwd, c_perplexity))
    print('nli: {}, ssn: {}, da: {}'.format(nli_std, ssn_std, da_std))
    out_f.close()
    json.dump(results, open('./data/result/result_{}.json'.format(experiment), 'w'), ensure_ascii=False)

def calc_average(y_true, y_pred):
    p = precision_score(y_true=y_true, y_pred=y_pred, average='macro')
    r = recall_score(y_true=y_true, y_pred=y_pred, average='macro')
    f = f1_score(y_true=y_true, y_pred=y_pred, average='macro')
    acc = accuracy_score(y_true=y_true, y_pred=y_pred)
    print('p: {} | r: {} | f: {} | acc: {}'.format(p, r, f, acc))


def save_cmx(y_true, y_pred, expr):
    fontsize = 40
    labels = sorted(list(set(y_true)))
    cmx_data = confusion_matrix(y_true, y_pred, labels=labels)

    df_cmx = pd.DataFrame(cmx_data, index=labels, columns=labels)

    plt.figure(figsize=(40, 30))
    plt.rcParams['font.size'] = fontsize
    heatmap = sns.heatmap(df_cmx, annot=True, fmt='d', cmap='Blues')
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig('./data/images/cmx_{}.png'.format(expr))


if __name__ == '__main__':
    args = parse()
    evaluate(args.expr)
