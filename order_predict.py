import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from nn_blocks import *
from train import initialize_env, parse
from utils import *
import time, random
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
from scipy import stats
from pprint import pprint
import copy


class OrderPredictor(nn.Module):
    def __init__(self, utterance_pair_encoder, da_encoder,
                 order_reasoning_layer, classifier, criterion, config):
        super(OrderPredictor, self).__init__()
        self.utterance_pair_encoder = utterance_pair_encoder
        self.da_encoder = da_encoder
        self.order_reasoning_layer = order_reasoning_layer
        self.classifier = classifier
        self.criterion = criterion
        self.config = config

    def forward(self, XTarget, DATarget, Y, step_size):
        """
        :param XOrdered: list of concated utterance pair tensors (3, batch_size, seq_len)
        :param XMisOrdered: list of concated utterance pair tensors (3, batch_size, seq_len)
        :param XTarget: list of concated utterance pair tensors (3, batch_size, seq_len)
        :param Y: label tensor which indicates ordered or misordered (batch_size, label_id)
        :param step_size: length of input minibatch (scalar)
        :param criterion: loss function
        :return: loss, predicted probability
        """
        # Encoding
        utterance_pair_hidden = self.utterance_pair_encoder.initHidden(step_size)
        for idx in range(len(XTarget)):
            t_output, t_hidden = self.utterance_pair_encoder(X=XTarget[idx], hidden=utterance_pair_hidden)
            XTarget[idx] = t_output
        XTarget = torch.stack(XTarget).squeeze(2)
        # Tensor:(window_size, batch_size, hidden_size)

        if self.config['use_da']:
            da_t_output = self.da_encoder(DATarget).permute(1,0,2)
        else:
            da_t_output = None
        # DATensor: (batch_size, window_size, hidden_size)
        XTarget, da_t_output = self.order_reasoning_layer(X=XTarget, DA=da_t_output,
                                                           hidden=self.order_reasoning_layer.initHidden(step_size),
                                                           da_hidden=self.order_reasoning_layer.initDAHidden(step_size))
        # output: (batch_size, hidden_size * 2)
        pred = self.classifier(XTarget, da_t_output).squeeze(1)
        Y = Y.squeeze(1)
        loss = self.criterion(pred, Y)
        torch.nn.utils.clip_grad_norm_(self.parameters(), self.config['clip'])
        if self.training:
            loss.backward()
        return loss.item(), pred.data.tolist()



def train(experiment):
    config = initialize_env(experiment)
    XD_train, YD_train, XU_train, YU_train = create_traindata(config=config, prefix='train')
    XD_valid, YD_valid, XU_valid, YU_valid = create_traindata(config=config, prefix='valid')
    if os.path.exists(os.path.join(config['log_root'], 'da_vocab.dict')):
        da_vocab = da_Vocab(config, create_vocab=False)
        utt_vocab = utt_Vocab(config, create_vocab=False)
    else:
        da_vocab = da_Vocab(config, das=[token for conv in XD_train + XD_valid + YD_train + YD_valid for token in conv])
        utt_vocab = utt_Vocab(config, sentences=[sentence for conv in XU_train + XU_valid + YU_train + YU_valid for sentence in conv])
        da_vocab.save()
        utt_vocab.save()
    filler_tag = ['acknowledge_(backchannel)', 'backchannel_in_question_form'] if config['lang'] == 'en' else ['フィラー', 'あいづち']
    XU_train, XD_train = zip(*[(conv, das) for conv, das in zip(XU_train, XD_train) if not das[-1] in filler_tag])
    XU_valid, XD_valid = zip(*[(conv, das) for conv, das in zip(XU_valid, XD_valid) if not das[-1] in filler_tag])
    XD_train = da_vocab.tokenize(XD_train)[:50000]
    XD_valid = da_vocab.tokenize(XD_valid)[:50000]
    XU_train = utt_vocab.tokenize(XU_train)[:50000]
    XU_valid = utt_vocab.tokenize(XU_valid)[:50000]
    XTarget, DATarget, Y = make_triple(XU_train, XD_train, config=config)
    print('Finish load data')

    lr = config['lr']
    batch_size = config['BATCH_SIZE']
    print_total_loss = 0

    utterance_encoder = UtteranceEncoder(utt_input_size=len(utt_vocab.word2id), embed_size=config['SSN_EMBED'],
                                              utterance_hidden=config['SSN_ENC_HIDDEN'], padding_idx=utt_vocab.word2id['<PAD>']).cuda()
    if config['use_da']:
        # da_pair_encoder = DAPairEncoder(da_hidden_size=config['SSN_DA_HIDDEN'], da_embed_size=config['SSN_DA_EMBED'], da_vocab_size=len(da_vocab.word2id)).cuda()
        da_encoder = DAEncoder(da_input_size=len(da_vocab.word2id), da_embed_size=config['SSN_DA_EMBED'], da_hidden=config['SSN_DA_HIDDEN'])
    else:
        da_encoder = None
    order_reasoning_layer = OrderReasoningLayer(encoder_hidden_size=config['SSN_ENC_HIDDEN'], hidden_size=config['SSN_REASONING_HIDDEN'],
                                                da_hidden_size=config['SSN_DA_HIDDEN'], attn=True).cuda()
    classifier = Classifier(hidden_size=config['SSN_REASONING_HIDDEN'] * 2, middle_layer_size=config['SSN_MIDDLE_LAYER'], da_hidden_size=config['SSN_DA_HIDDEN']).cuda()
    predictor = OrderPredictor(utterance_pair_encoder=utterance_encoder, order_reasoning_layer=order_reasoning_layer,
                               da_encoder=da_encoder, classifier=classifier, criterion=nn.BCELoss(), config=config).cuda()
    model_opt = optim.Adam(predictor.parameters(), lr=lr, weight_decay=1e-5)
    print('total parameters: ', count_parameters(predictor))

    print('--- Start Training ---')
    start = time.time()
    _valid_loss = None
    _train_loss = None
    early_stop = 0
    indexes = [i for i in range(len(XTarget))]
    for e in range(config['EPOCH']):
        tmp_time = time.time()
        print('Epoch {} start'.format(e+1))
        random.shuffle(indexes)
        k = 0
        predictor.train()
        train_acc = []
        while k < len(indexes):
            step_size = min(batch_size, len(indexes)-k)
            print('\rTRAINING|\t{} / {}'.format(k + step_size, len(indexes)), end='')
            batch_idx = indexes[k: k+step_size]
            model_opt.zero_grad()

            Xtarget = padding([XTarget[i] for i in batch_idx], pad_idx=utt_vocab.word2id['<PAD>'])
            if config['use_da']:
                DAtarget = torch.tensor([DATarget[i] for i in batch_idx]).cuda()
            else:
                DAtarget = None
            y = torch.tensor([Y[i] for i in batch_idx], dtype=torch.float).cuda()
            loss, pred = predictor(XTarget=Xtarget, DATarget=DAtarget, Y=y, step_size=step_size)
            print_total_loss += loss
            model_opt.step()
            k += step_size
            assert not all(ele == pred[0] for ele in pred[1:]), 'All probability is same value'
            result = [0 if line < 0.5 else 1 for line in pred]
            train_acc.append(accuracy_score(y_true=y.data.tolist(), y_pred=result))
        print()
        valid_loss, valid_acc = validation(XU_valid=XU_valid, XD_valid=XD_valid,
                                           model=predictor, utt_vocab=utt_vocab, config=config)

        def save_model(filename):
            torch.save(predictor.state_dict(), os.path.join(config['log_dir'], 'orderpred_state{}.model'.format(filename)))

        if _valid_loss is None:
            save_model('validbest')
            _valid_loss = valid_acc
        else:
            if _valid_loss < valid_acc:
                save_model('validbest')
                _valid_loss = valid_acc
                print('valid loss update, save model')

        if _train_loss is None:
            save_model('trainbest')
            _train_loss = print_total_loss
        else:
            if _train_loss > print_total_loss:
                save_model('trainbest')
                _train_loss = print_total_loss
                early_stop = 0
                print('train loss update, save model')
            else:
                early_stop += 1
                print('early stopping count | {} / {}'.format(early_stop, config['EARLY_STOP']))
                if early_stop >= config['EARLY_STOP']:
                    break
        if (e + 1) % config['LOGGING_FREQ'] == 0:
            print_loss_avg = print_total_loss / config['LOGGING_FREQ']
            print_total_loss = 0
            print('train acc. | ', np.mean(train_acc))
            print('epoch %d\tloss %.4f\tvalid loss %.4f\t | exec time %.4f' % (e + 1, print_loss_avg, valid_loss, time.time() - tmp_time))

        if (e + 1) % config['SAVE_MODEL'] == 0:
            print('save model')
            save_model(e+1)

    print()
    print('Finish training | exec time: %.4f [sec]' % (time.time() - start))


def validation(XU_valid, XD_valid, model, utt_vocab, config):
    model.eval()
    XTarget, DATarget, Y = make_triple(XU_valid, XD_valid, config=config)
    indexes = [i for i in range(len(XTarget))]
    random.shuffle(indexes)
    k = 0
    total_loss = 0
    valid_acc = []
    while k < len(indexes):
        step_size = min(config['BATCH_SIZE'], len(indexes)-k)
        # print('\rVALIDATION|\t{} / {} steps'.format(k + step_size, len(indexes)), end='')
        batch_idx = indexes[k : k+step_size]
        Xtarget = padding([XTarget[i] for i in batch_idx], pad_idx=utt_vocab.word2id['<PAD>'])
        if config['use_da']:
            DAtarget = torch.tensor([DATarget[i] for i in batch_idx]).cuda()
        else:
            DAtarget = None
        y = torch.tensor([Y[i] for i in batch_idx], dtype=torch.float).cuda()
        loss, preds = model.forward(XTarget=Xtarget, DATarget=DAtarget, Y=y, step_size=step_size)
        result = [0 if line < 0.5 else 1 for line in preds]
        valid_acc.append(accuracy_score(y_true=y.data.tolist(), y_pred=result))
        k += step_size
        total_loss += loss
    print('avg. of valid acc:\t{}'.format(np.mean(valid_acc)))
    return total_loss, np.mean(valid_acc)


def evaluate(experiment):
    config = initialize_env(experiment)
    XD_test, YD_test, XU_test, YU_test = create_traindata(config=config, prefix='test')
    da_vocab = da_Vocab(config=config, create_vocab=False)
    utt_vocab = utt_Vocab(config=config, create_vocab=False)
    XD_test = da_vocab.tokenize(XD_test)
    XU_test = utt_vocab.tokenize(XU_test)
    XTarget, DATarget, Y = make_triple(XU_test, XD_test, config=config)

    utterance_pair_encoder = UtteranceEncoder(utt_input_size=len(utt_vocab.word2id), embed_size=config['SSN_EMBED'],
                                              utterance_hidden=config['SSN_ENC_HIDDEN'], padding_idx=utt_vocab.word2id['<PAD>']).cuda()
    if config['use_da']:
        # da_pair_encoder = DAPairEncoder(da_hidden_size=config['SSN_DA_HIDDEN'], da_embed_size=config['SSN_DA_EMBED'], da_vocab_size=len(da_vocab.word2id)).cuda()
        da_encoder = DAEncoder(da_input_size=len(da_vocab.word2id), da_embed_size=config['SSN_DA_EMBED'], da_hidden=config['SSN_DA_HIDDEN'])
    else:
        da_encoder = None
    order_reasoning_layer = OrderReasoningLayer(encoder_hidden_size=config['SSN_ENC_HIDDEN'], hidden_size=config['SSN_REASONING_HIDDEN'],
                                                da_hidden_size=config['SSN_DA_HIDDEN'], attn=True).cuda()
    classifier = Classifier(hidden_size=config['SSN_REASONING_HIDDEN'] * 2, middle_layer_size=config['SSN_MIDDLE_LAYER'], da_hidden_size=config['SSN_DA_HIDDEN'])
    predictor = OrderPredictor(utterance_pair_encoder=utterance_pair_encoder, order_reasoning_layer=order_reasoning_layer,
                               da_encoder=da_encoder, classifier=classifier, criterion=nn.BCELoss(), config=config).cuda()
    predictor.load_state_dict(torch.load(os.path.join(config['log_dir'], 'orderpred_statevalidbest.model')))

    predictor.eval()
    acc = []
    indexes = [i for i in range(len(XTarget))]
    for _ in range(5):
        y_preds = []
        y_trues = []
        k = 0
        while k < len(indexes):
            step_size = min(config['BATCH_SIZE'], len(indexes) - k)
            batch_idx = indexes[k : k + step_size]
            Xtarget = padding([XTarget[i] for i in batch_idx], pad_idx=utt_vocab.word2id['<PAD>'])
            if config['use_da']:
                DAtarget = torch.tensor([DATarget[i] for i in batch_idx]).cuda()
            else:
                DAtarget = None
            y = torch.tensor([Y[i] for i in batch_idx], dtype=torch.float).cuda()
            loss, preds = predictor.forward(XTarget=Xtarget, DATarget=DAtarget, Y=y, step_size=step_size)
            result = [0 if line < 0.5 else 1 for line in preds]
            y_preds.extend(result)
            y_trues.extend(y.data.tolist())
            k += step_size
        acc.append(accuracy_score(y_pred=y_preds, y_true=y_trues))
    acc = np.array(acc)
    print('Avg. of Accuracy: {} +- {}'.format(acc.mean(), conf_interval(acc)))


def conf_interval(X):
    t = 2.1318
    loc = X.mean()
    scale = np.sqrt(X.var()/5)
    return t * scale

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def make_triple(utterance_pairs, da_pairs, config={'m':5}):
    Xtarget = []
    DAtarget = []
    Y = []
    for bidx in range(len(utterance_pairs)):
        da_seq = da_pairs[bidx]
        for _ in range(config['m']):
            target, da_target, label = sample_triple(utterance_pairs[bidx], da_seq)
            Xtarget.append(target)
            DAtarget.append(da_target)
            Y.append(label)
    return Xtarget, DAtarget, Y

def sample_triple(pairs, da_pairs):
    utterances = copy.copy(pairs)
    das = copy.copy(da_pairs)
    if random.random() > 0.5:
        label = [0]
    else:
        sampled_idx = random.sample([i for i in range(len(pairs) - 1)], 1)[0]
        tmp = utterances[sampled_idx]
        utterances[sampled_idx] = utterances[-1]
        utterances[-1] = tmp
        tmp = das[sampled_idx]
        das[sampled_idx] = das[-1]
        das[-1] = tmp
        label = [1]
    return utterances, das, label


def padding(batch, pad_idx):
    pair_list = []
    for i in range(len(batch[0])):
        max_seq_len = max(len(b[i]) + 1 for b in batch)
        for ci in range(len(batch)):
            batch[ci][i] = batch[ci][i] + [pad_idx] * (max_seq_len - len(batch[ci][i]))
        pair_list.append(torch.tensor([b[i] for b in batch]).cuda())
    return pair_list


if __name__ == '__main__':
    args = parse()
    train(args.expr)
    evaluate(args.expr)
