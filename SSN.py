import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from nn_blocks import *
from train import initialize_env, make_batchidx, parse
from utils import *
import time, random
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
from scipy import stats
from pprint import pprint


class OrderPredictor(nn.Module):
    def __init__(self, utterance_pair_encoder, da_pair_encoder,
                 order_reasoning_layer, classifier, config):
        super(OrderPredictor, self).__init__()
        self.utterance_pair_encoder = utterance_pair_encoder
        self.da_pair_encoder = da_pair_encoder
        self.order_reasoning_layer = order_reasoning_layer
        self.classifier = classifier
        self.config = config

    def forward(self, XOrdered, XMisOrdered, XTarget,
                DAOrdered, DAMisOrdered, DATarget,
                Y, step_size, criterion):
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
        # utterance_pair_encoder_param = list(self.utterance_pair_encoder.named_parameters())
        utterance_pair_hidden = self.utterance_pair_encoder.initHidden(step_size)
        for idx in range(len(XOrdered)):
            o_output, _ = self.utterance_pair_encoder(X=XOrdered[idx], hidden=utterance_pair_hidden)
            XOrdered[idx] = o_output
            m_output, _ = self.utterance_pair_encoder(X=XMisOrdered[idx], hidden=utterance_pair_hidden)
            XMisOrdered[idx] = m_output
        for idx in range(len(XTarget)):
            t_output, t_hidden = self.utterance_pair_encoder(X=XTarget[idx], hidden=utterance_pair_hidden)
            XTarget[idx] = t_output
        XOrdered = torch.stack(XOrdered).squeeze(2)
        XMisOrdered = torch.stack(XMisOrdered).squeeze(2)
        XTarget = torch.stack(XTarget).squeeze(2)
        # utt_pair_dot = make_dot(XOrdered.view(-1).sum(), params=dict(utterance_pair_encoder_param))
        # utt_pair_dot.render("./data/images/utt_pair_enc")
        # print('A: {}, {}, {}'.format(XOrdered.grad, XMisOrdered.grad, XTarget.grad))
        # Tensor:(window_size, batch_size, hidden_size)

        if self.config['use_da']:
            da_o_output = self.da_pair_encoder(DAOrdered).permute(1,0,2)
            da_m_output = self.da_pair_encoder(DAMisOrdered).permute(1,0,2)
            da_t_output = self.da_pair_encoder(DATarget).permute(1,0,2)
        else:
            da_o_output, da_m_output, da_t_output = None, None, None
        # DATensor: (batch_size, window_size, hidden_size)
        # order_reasoning_layer_param = list(self.order_reasoning_layer.named_parameters()) + utterance_pair_encoder_param + list(self.named_parameters())
        XOrdered, da_o_output = self.order_reasoning_layer(X=XOrdered, DA=da_o_output,
                                                           hidden=self.order_reasoning_layer.initHidden(step_size),
                                                           da_hidden=self.order_reasoning_layer.initDAHidden(step_size))
        XMisOrdered, da_m_output = self.order_reasoning_layer(X=XMisOrdered, DA=da_m_output,
                                                           hidden=self.order_reasoning_layer.initHidden(step_size),
                                                           da_hidden=self.order_reasoning_layer.initDAHidden(step_size))
        XTarget, da_t_output = self.order_reasoning_layer(X=XTarget, DA=da_t_output,
                                                           hidden=self.order_reasoning_layer.initHidden(step_size),
                                                           da_hidden=self.order_reasoning_layer.initDAHidden(step_size))
        # orl_dot = make_dot(XOrdered.view(-1).sum(), params=dict(order_reasoning_layer_param))
        # orl_dot.render("./data/images/orl")
        # print('B: {}, {}, {}'.format(XOrdered.grad, XMisOrdered.grad, XTarget.grad))
        if not da_o_output is None:
            da_output = torch.cat((da_o_output, da_m_output, da_t_output), dim=-1)
            # da_output: (batch_size, da_hidden_size * 3)
        else:
            da_output = None
        output = torch.cat((XOrdered, XMisOrdered, XTarget), dim=-1)

        # print('C: ', output.grad)
        # output: (batch_size, hidden_size * 6)
        # classifier_param = list(self.classifier.named_parameters()) + order_reasoning_layer_param
        pred = self.classifier(output, da_output).squeeze(1)
        Y = Y.squeeze(1)
        loss = criterion(pred, Y)
        torch.nn.utils.clip_grad_norm_(self.parameters(), self.config['clip'])
        # classifier_dot = make_dot(loss, params=classifier_param)
        # classifier_dot.render("./data/images/classifier")
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
        da_vocab = da_Vocab(config, XD_train + XD_valid, YD_train + YD_valid)
        utt_vocab = utt_Vocab(config, XU_train + XU_valid, YU_train + YU_valid)
        da_vocab.save()
        utt_vocab.save()
    XD_train, YD_train = da_vocab.tokenize(XD_train, YD_train)
    XD_valid, YD_valid = da_vocab.tokenize(XD_valid, YD_valid)
    XU_train, YU_train = utt_vocab.tokenize(XU_train, YU_train)
    XU_valid, YU_valid = utt_vocab.tokenize(XU_valid, YU_valid)
    print('Finish load vocab')

    lr = config['lr']
    batch_size = config['BATCH_SIZE']
    print_total_loss = 0

    utterance_pair_encoder = UtteranceEncoder(utt_input_size=len(utt_vocab.word2id), embed_size=config['SSN_EMBED'],
                                              utterance_hidden=config['SSN_ENC_HIDDEN'], padding_idx=utt_vocab.word2id['<PAD>']).cuda()
    if config['use_da']:
        da_pair_encoder = DAPairEncoder(da_hidden_size=config['SSN_DA_HIDDEN'], da_embed_size=config['SSN_DA_EMBED'], da_vocab_size=len(da_vocab.word2id)).cuda()
        # da_pair_encoder_opt = optim.Adam(da_pair_encoder.parameters(), lr=lr)
    else:
        da_pair_encoder = None
    order_reasoning_layer = OrderReasoningLayer(encoder_hidden_size=config['SSN_ENC_HIDDEN'], hidden_size=config['SSN_REASONING_HIDDEN'],
                                                da_hidden_size=config['SSN_DA_HIDDEN']).cuda()
    classifier = Classifier(hidden_size=config['SSN_REASONING_HIDDEN'], middle_layer_size=config['SSN_MIDDLE_LAYER'], da_hidden_size=config['SSN_DA_HIDDEN'])
    # utterance_pair_encoder_opt = optim.Adam(utterance_pair_encoder.parameters(), lr=lr)
    # order_reasoning_layer_opt = optim.Adam(order_reasoning_layer.parameters(), lr=lr)
    # classifier_opt = optim.Adam(classifier.parameters(), lr=lr)

    predictor = OrderPredictor(utterance_pair_encoder=utterance_pair_encoder, order_reasoning_layer=order_reasoning_layer,
                               da_pair_encoder=da_pair_encoder, classifier=classifier, config=config).cuda()
    # pprint(list(predictor.named_parameters()))
    model_opt = optim.Adam(predictor.parameters(), lr=lr, weight_decay=1e-5)
    print('total parameters: ', count_parameters(predictor))
    # criterion = nn.CrossEntropyLoss(ignore_index=utt_vocab.word2id['<PAD>'])
    criterion = nn.BCELoss()
    print('--- Start Training ---')
    start = time.time()
    _valid_loss = None
    _train_loss = None
    early_stop = 0
    utterance_pairs = [[XU + [utt_vocab.word2id['<SEP>']] + YU for XU, YU in zip(XU_train[conv_idx], YU_train[conv_idx])] for conv_idx in range(len(XU_train))]
    utterance_pairs = [[batch[pi] for pi in range(0, len(batch), 2)] for batch in utterance_pairs]
    if config['use_da']:
        da_pairs = [[[XD, YD] for XD, YD in zip(XD_train[conv_idx], YD_train[conv_idx])] for conv_idx in range(len(XD_train))]
        da_pairs = [[batch[pi] for pi in range(0, len(batch), 2)] for batch in da_pairs]
    else:
        da_pairs = None

    (XOrdered, XMisOrdered, XTarget), (DAOrdered, DAMisOrdered, DATarget), Y = make_triple(utterance_pairs=utterance_pairs, da_pairs=da_pairs)

    for e in range(config['EPOCH']):
        tmp_time = time.time()
        print('Epoch {} start'.format(e+1))
        indexes = [i for i in range(len(XOrdered))]
        random.shuffle(indexes)
        k = 0
        predictor.train()
        train_acc = []
        while k < len(indexes):
            step_size = min(batch_size, len(indexes)-k)
            print('\rTRAINING|\t{} / {}'.format(k + step_size, len(indexes)), end='')
            batch_idx = indexes[k: k+step_size]
            model_opt.zero_grad()

            Xordered = padding([XOrdered[i] for i in batch_idx], pad_idx=utt_vocab.word2id['<PAD>'])
            Xmisordered = padding([XMisOrdered[i] for i in batch_idx], pad_idx=utt_vocab.word2id['<PAD>'])
            Xtarget = padding([XTarget[i] for i in batch_idx], pad_idx=utt_vocab.word2id['<PAD>'])
            if config['use_da']:
                DAordered = torch.tensor([DAOrdered[i] for i in batch_idx]).cuda()
                DAmisordered = torch.tensor([DAMisOrdered[i] for i in batch_idx]).cuda()
                DAtarget = torch.tensor([DATarget[i] for i in batch_idx]).cuda()
            else:
                DAordered, DAmisordered, DAtarget = None, None, None
            y = torch.tensor([Y[i] for i in batch_idx], dtype=torch.float).cuda()
            loss, pred = predictor(XOrdered=Xordered, XMisOrdered=Xmisordered, XTarget=Xtarget,
                                   DAOrdered=DAordered, DAMisOrdered=DAmisordered, DATarget=DAtarget,
                                   Y=y, step_size=step_size, criterion=criterion)
            print_total_loss += loss
            model_opt.step()
            k += step_size
            assert not all(ele == pred[0] for ele in pred[1:]), 'All probability is same value'
            result = [0 if line < 0.5 else 1 for line in pred]
            train_acc.append(accuracy_score(y_true=y.data.tolist(), y_pred=result))
        print()
        valid_loss = validation(experiment=experiment, XU_valid=XU_train, YU_valid=YU_train, XD_valid=XD_train, YD_valid=YD_train,
                                model=predictor, utt_vocab=utt_vocab, config=config)

        def save_model(filename):
            torch.save(utterance_pair_encoder.state_dict(), os.path.join(config['log_dir'], 'utt_pair_enc_state{}.model'.format(filename)))
            torch.save(order_reasoning_layer.state_dict(), os.path.join(config['log_dir'], 'ord_rsn_state{}.model'.format(filename)))
            torch.save(classifier.state_dict(), os.path.join(config['log_dir'], 'cls_state{}.model'.format(filename)))
            if config['use_da']:
                torch.save(da_pair_encoder.state_dict(), os.path.join(config['log_dir'], 'da_pair_enc_state{}.model'.format(filename)))

        if _valid_loss is None:
            save_model('validbest')
            _valid_loss = valid_loss
        else:
            if _valid_loss > valid_loss:
                save_model('validbest')
                _valid_loss = valid_loss
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


def validation(experiment, XU_valid, YU_valid, XD_valid, YD_valid, model, utt_vocab, config):
    model.eval()
    indexes = [i for i in range(len(XU_valid))]
    random.shuffle(indexes)
    criterion = nn.BCELoss()
    k = 0
    total_loss = 0
    valid_acc = []
    while k < len(indexes):
        step_size = min(config['BATCH_SIZE'], len(indexes)-k)
        # print('\rVALIDATION|\t{} / {} steps'.format(k + step_size, len(indexes)), end='')
        batch_idx = indexes[k : k+step_size]
        utterance_pairs = [[XU + [utt_vocab.word2id['<SEP>']] + YU for XU, YU in zip(XU_valid[seq_idx], YU_valid[seq_idx])] for seq_idx in batch_idx]
        if config['use_da']:
            da_pairs = [[[XD, YD] for XD, YD in zip(XD_valid[seq_idx], YD_valid[seq_idx])] for seq_idx in batch_idx]
        else:
            da_pairs = None

        (Xordered, Xmisordered, Xtarget), (DAordered, DAmisordered, DAtarget), Y = make_triple(utterance_pairs, da_pairs)
        Xordered = padding(Xordered, pad_idx=utt_vocab.word2id['<PAD>'])
        Xmisordered = padding(Xmisordered, pad_idx=utt_vocab.word2id['<PAD>'])
        Xtarget = padding(Xtarget, pad_idx=utt_vocab.word2id['<PAD>'])
        if config['use_da']:
            DAordered = torch.tensor(DAordered).cuda()
            DAmisordered = torch.tensor(DAmisordered).cuda()
            DAtarget = torch.tensor(DAtarget).cuda()
        else:
            DAordered, DAmisordered, DAtarget = None, None, None
        y = torch.tensor(Y, dtype=torch.float).cuda()
        loss, preds = model.forward(XOrdered=Xordered, XMisOrdered=Xmisordered, XTarget=Xtarget,
                                    DAOrdered=DAordered, DAMisOrdered=DAmisordered, DATarget=DAtarget,
                                    Y=y, step_size=step_size*5, criterion=criterion)
        result = [0 if line < 0.5 else 1 for line in preds]
        valid_acc.append(accuracy_score(y_true=y.data.tolist(), y_pred=result))
        k += step_size
        total_loss += loss
    print('avg. of valid acc:\t{}'.format(np.mean(valid_acc)))
    return total_loss


def evaluate(experiment):
    config = initialize_env(experiment)
    XD_test, YD_test, XU_test, YU_test = create_traindata(config=config, prefix='test')
    da_vocab = da_Vocab(config=config, create_vocab=False)
    utt_vocab = utt_Vocab(config=config, create_vocab=False)
    XD_test, YD_test = da_vocab.tokenize(XD_test, YD_test)
    XU_test, YU_test = utt_vocab.tokenize(XU_test, YU_test)

    utterance_pair_encoder = UtteranceEncoder(utt_input_size=len(utt_vocab.word2id), embed_size=config['SSN_EMBED'],
                                              utterance_hidden=config['SSN_ENC_HIDDEN'], padding_idx=utt_vocab.word2id['<PAD>']).cuda()
    utterance_pair_encoder.load_state_dict(torch.load(os.path.join(config['log_dir'], 'utt_pair_enc_statevalidbest.model')))
    if config['use_da']:
        da_pair_encoder = DAPairEncoder(da_hidden_size=config['SSN_DA_HIDDEN'], da_embed_size=config['SSN_DA_EMBED'], da_vocab_size=len(da_vocab.word2id)).cuda()
        da_pair_encoder.load_state_dict(torch.load(os.path.join(config['log_dir'], 'da_pair_enc_statevalidbest.model')))
    else:
        da_pair_encoder = None
    order_reasoning_layer = OrderReasoningLayer(encoder_hidden_size=config['SSN_ENC_HIDDEN'], hidden_size=config['SSN_REASONING_HIDDEN'],
                                                da_hidden_size=config['SSN_DA_HIDDEN']).cuda()
    classifier = Classifier(hidden_size=config['SSN_REASONING_HIDDEN'], middle_layer_size=config['SSN_MIDDLE_LAYER'], da_hidden_size=config['SSN_DA_HIDDEN'])
    order_reasoning_layer.load_state_dict(torch.load(os.path.join(config['log_dir'], 'ord_rsn_statevalidbest.model')))
    classifier.load_state_dict(torch.load(os.path.join(config['log_dir'], 'cls_statevalidbest.model')))
    predictor = OrderPredictor(utterance_pair_encoder=utterance_pair_encoder, order_reasoning_layer=order_reasoning_layer,
                               da_pair_encoder=da_pair_encoder, classifier=classifier, config=config).cuda()

    criterion = nn.BCELoss()
    predictor.eval()
    k = 0
    y_preds = []
    y_trues = []
    acc = []
    indexes = [i for i in range(len(XU_test))]
    for _ in range(5):
        while k < len(indexes):
            step_size = min(config['BATCH_SIZE'], len(indexes) - k)
            batch_idx = indexes[k : k + step_size]
            utterance_pairs = [[XU + [utt_vocab.word2id['<SEP>']] + YU for XU, YU in zip(XU_test[seq_idx], YU_test[seq_idx])] for seq_idx in batch_idx]
            if config['use_da']:
                da_pairs = [[[XD, YD] for XD, YD in zip(XD_test[seq_idx], YD_test[seq_idx])] for seq_idx in batch_idx]
            else:
                da_pairs = None
            if experiment == 'baseline':
                (Xordered, Xmisordered, Xtarget), (DAordered, DAmisordered, DAtarget), Y = baseline_triples(utterance_pairs, da_pairs)
            else:
                (Xordered, Xmisordered, Xtarget), (DAordered, DAmisordered, DAtarget), Y = make_triple(utterance_pairs, da_pairs)
            Xordered = padding(Xordered, pad_idx=utt_vocab.word2id['<PAD>'])
            Xmisordered = padding(Xmisordered, pad_idx=utt_vocab.word2id['<PAD>'])
            Xtarget = padding(Xtarget, pad_idx=utt_vocab.word2id['<PAD>'])
            if config['use_da']:
                DAordered = torch.tensor(DAordered).cuda()
                DAmisordered = torch.tensor(DAmisordered).cuda()
                DAtarget = torch.tensor(DAtarget).cuda()
            else:
                DAordered, DAmisordered, DAtarget = None, None, None
            y = torch.tensor(Y, dtype=torch.float).cuda()
            if experiment == 'baseline':
                loss, preds = predictor.baseline(XOrdered=Xordered, XTarget=Xtarget,
                                             DAOrdered=DAordered, DATarget=DAtarget,
                                             Y=y, step_size=step_size, criterion=criterion)

            else:
                loss, preds = predictor.forward(XOrdered=Xordered, XMisOrdered=Xmisordered, XTarget=Xtarget,
                                            DAOrdered=DAordered, DAMisOrdered=DAmisordered, DATarget=DAtarget,
                                            Y=y, step_size=step_size, criterion=criterion)
            result = [0 if line < 0.5 else 1 for line in preds]
            y_preds.extend(result)
            y_trues.extend(y.data.tolist())
            k += step_size
        acc.append(accuracy_score(y_pred=y_preds, y_true=y_trues))
    acc = np.array(acc)
    # t_dist = stats.t(loc=acc.mean(), scale=np.sqrt(acc.var()/len(acc)), df=len(acc)-1)
    # bottom, up = t_dist.interval(alpha=0.95)
    print('Avg. of Accuracy: {} +- {}'.format(acc.mean(), conf_interval(acc)))


def conf_interval(X):
    t = 2.1318
    loc = X.mean()
    scale = np.sqrt(X.var()/5)
    return t * scale

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def make_triple(utterance_pairs, da_pairs=None):
    Xordered = []
    Xmisordered = []
    Xtarget = []
    DAordered = []
    DAmisordered = []
    DAtarget = []
    Y = []
    for bidx in range(len(utterance_pairs)):
        if not da_pairs is None:
            da_seq = da_pairs[bidx]
        else:
            da_seq = None
        for _ in range(5):
            (ordered, misordered, target), (da_ordered, da_misordered, da_target), label = sample_triple(utterance_pairs[bidx], da_seq)
            Xordered.append(ordered)
            Xmisordered.append(misordered)
            Xtarget.append(target)
            DAordered.append(da_ordered)
            DAmisordered.append(da_misordered)
            DAtarget.append(da_target)
            Y.append(label)
    # padding
    # Xordered = padding(Xordered, utt_vocab.word2id['<PAD>'])
    # Xmisordered = padding(Xmisordered, utt_vocab.word2id['<PAD>'])
    # Xtarget = padding(Xtarget, utt_vocab.word2id['<PAD>'])
    # if not da_pairs is None:
    #     da_ordered = torch.tensor(DAordered).cuda()
    #     da_misordered = torch.tensor(DAmisordered).cuda()
    #     da_target = torch.tensor(DAtarget).cuda()
    return (Xordered, Xmisordered, Xtarget), (DAordered, DAmisordered, DAtarget), Y

def baseline_triples(utterance_pairs, da_pairs=None):
    Xordered = []
    Xmisordered = []
    Xtarget = []
    DAordered = []
    DAmisordered = []
    DAtarget = []
    Y = []
    for bidx in range(len(utterance_pairs)):
        if not da_pairs is None:
            da_seq = da_pairs[bidx]
            DAordered.append(da_seq[:-1])
        else:
            da_seq = None
        (ordered, misordered, target), (da_ordered, da_misordered, da_target), label = sample_triple(utterance_pairs[bidx], da_seq)
        Xordered.append(utterance_pairs[bidx][:-1])
        Xmisordered.append(misordered)
        Xtarget.append(target)
        DAmisordered.append(da_misordered)
        DAtarget.append(da_target)
        Y.append(label)
    return (Xordered, Xmisordered, Xtarget), (DAordered, DAmisordered, DAtarget), Y

def sample_triple(pairs, da_pairs):
    sampled_idx = random.sample([i for i in range(len(pairs)-1)], 3)
    i, j, k = sorted(sampled_idx)
    # i, j, k = -4, -3, -2
    ordered = [pairs[i], pairs[j], pairs[k]]
    misordered = [pairs[i], pairs[k], pairs[j]]
    if da_pairs is None:
        da_ordered = None
        da_misordered = None
    else:
        da_ordered = [da_pairs[i], da_pairs[j], da_pairs[k]]
        da_misordered = [da_pairs[i], da_pairs[k], da_pairs[j]]
    if random.random() > 0.5:
        target = [pairs[j], pairs[k], pairs[-1]]
        if da_pairs is None:
            da_target = None
        else:
            da_target = [da_pairs[j], da_pairs[k], da_pairs[-1]]
        label = [0]
    else:
        target = [pairs[j], pairs[-1], pairs[k]]
        if da_pairs is None:
            da_target = None
        else:
            da_target = [da_pairs[j], da_pairs[-1], da_pairs[k]]
        label = [1]
    return (ordered, misordered, target), (da_ordered, da_misordered, da_target), label


def padding(batch, pad_idx):
    pair_list = []
    for i in range(len(batch[0])):
        max_seq_len = max(len(b[i]) + 1 for b in batch)
        for ci in range(len(batch)):
            batch[ci][i] = batch[ci][i] + [pad_idx] * (max_seq_len - len(batch[ci][i]))
        pair_list.append(torch.tensor([b[i] for b in batch]).cuda())
    return pair_list


if __name__ == '__main__':
    global args
    args = parse()

    train(args.expr)
    evaluate(args.expr)
