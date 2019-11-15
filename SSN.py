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

class OrderPredictor(nn.Module):
    def __init__(self, utterance_pair_encoder, da_pair_encoder,
                 order_reasoning_layer, config, device='cpu'):
        super(OrderPredictor, self).__init__()
        self.utterance_pair_encoder = utterance_pair_encoder
        self.da_pair_encoder = da_pair_encoder
        self.order_reasoning_layer = order_reasoning_layer
        self.config = config
        self.device = device

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
        utterance_pair_hidden = self.utterance_pair_encoder.initHidden(step_size, self.device)
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
        # Tensor:(window_size, batch_size, hidden_size)

        if self.config['use_da']:
            da_o_output = self.da_pair_encoder(DAOrdered).permute(1,0,2)
            da_m_output = self.da_pair_encoder(DAMisOrdered).permute(1,0,2)
            da_t_output = self.da_pair_encoder(DATarget).permute(1,0,2)
        else:
            da_o_output, da_m_output, da_t_output = None, None, None
        # DATensor: (batch_size, window_size, hidden_size)

        pred = self.order_reasoning_layer(XOrdered=XOrdered, XMisOrdered=XMisOrdered, XTarget=XTarget,
                                          DAOrdered=da_o_output, DAMisOrdered=da_m_output, DATarget=da_t_output,
                                          Y=Y, hidden=self.order_reasoning_layer.initHidden(step_size, self.device),
                                          da_hidden=self.order_reasoning_layer.initDAHidden(step_size, self.device))
        Y = Y.squeeze(1)
        loss = criterion(pred, Y)
        if self.training:
            loss.backward()
        return loss.item(), pred.data.tolist()

def train(experiment):
    config = initialize_env(experiment)
    XD_train, YD_train, XU_train, YU_train = create_traindata(config=config, prefix='train')
    XD_valid, YD_valid, XU_valid, YU_valid = create_traindata(config=config, prefix='dev')
    if os.path.exists(os.path.join(config['log_root'], 'da_vocab.dict')):
        da_vocab = da_Vocab(config, create_vocab=False)
        utt_vocab = utt_Vocab(config, create_vocab=False)
    else:
        da_vocab = da_Vocab(config, XD_train + XD_valid, YD_train, YD_valid)
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
                                              utterance_hidden=config['SSN_ENC_HIDDEN'], padding_idx=utt_vocab.word2id['<PAD>']).to(device)
    if config['use_da']:
        da_pair_encoder = DAPairEncoder(da_hidden_size=config['SSN_DA_HIDDEN'], da_embed_size=config['SSN_DA_EMBED'], da_vocab_size=len(da_vocab.word2id)).to(device)
        da_pair_encoder_opt = optim.Adam(da_pair_encoder.parameters(), lr=lr)
    else:
        da_pair_encoder = None
    order_reasoning_layer = OrderReasoningLayer(encoder_hidden_size=config['SSN_ENC_HIDDEN'], hidden_size=config['SSN_REASONING_HIDDEN'],
                                                middle_layer_size=config['SSN_MIDDLE_LAYER'], da_hidden_size=config['SSN_DA_HIDDEN'], config=config).to(device)
    utterance_pair_encoder_opt = optim.Adam(utterance_pair_encoder.parameters(), lr=lr)
    order_reasoning_layer_opt = optim.Adam(order_reasoning_layer.parameters(), lr=lr)


    predictor = OrderPredictor(utterance_pair_encoder=utterance_pair_encoder, order_reasoning_layer=order_reasoning_layer,
                               da_pair_encoder=da_pair_encoder, config=config, device=device).to(device)
    criterion = nn.CrossEntropyLoss()
    print('--- Start Training ---')
    start = time.time()
    _valid_loss = None
    _train_loss = None
    early_stop = 0

    for e in range(config['EPOCH']):
        tmp_time = time.time()
        print('Epoch {} start'.format(e+1))

        indexes = [i for i in range(len(XU_train))]
        random.shuffle(indexes)
        k = 0
        predictor.train()
        while k < len(indexes):
            step_size = min(batch_size, len(indexes)-k)
            batch_idx = indexes[k: k+step_size]
            utterance_pair_encoder_opt.zero_grad()
            order_reasoning_layer_opt.zero_grad()
            print('\rTRAINING|\t{} / {}'.format(k + step_size, len(indexes)), end='')

            utterance_pairs = [[XU + [utt_vocab.word2id['<SEP>']] + YU for XU, YU in zip(XU_train[seq_idx], YU_train[seq_idx])] for seq_idx in batch_idx]
            if config['use_da']:
                da_pairs = [[[XD, YD] for XD, YD in zip(XD_train[seq_idx], YD_train[seq_idx])] for seq_idx in batch_idx]
            else:
                da_pairs = None
            # utterance_pairs: (batch_size, conv_len, seq_len)
            (Xordered, Xmisordered, Xtarget), (DAordered, DAmisordered, DAtarget), Y = make_triple(utterance_pairs, utt_vocab, da_pairs)
            loss, pred = predictor.forward(XOrdered=Xordered, XMisOrdered=Xmisordered, XTarget=Xtarget,
                                           DAOrdered=DAordered, DAMisOrdered=DAmisordered, DATarget=DAtarget,
                                           Y=Y, step_size=step_size, criterion=criterion)
            print_total_loss += loss
            utterance_pair_encoder_opt.step()
            order_reasoning_layer_opt.step()
            k += step_size
        print()
        valid_loss = validation(XU_valid=XU_valid, YU_valid=YU_valid, XD_valid=XD_valid, YD_valid=YD_valid,
                                model=predictor, utt_vocab=utt_vocab, config=config)

        def save_model(filename):
            torch.save(utterance_pair_encoder.state_dict(), os.path.join(config['log_dir'], 'utt_pair_enc_state{}.model'.format(filename)))
            torch.save(order_reasoning_layer.state_dict(), os.path.join(config['log_dir'], 'ord_rsn_state{}.model'.format(filename)))
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
            print('epoch %d\tloss %.4f\tvalid loss %.4f\t | exec time %.4f' % (e + 1, print_loss_avg, valid_loss, time.time() - tmp_time))

        if (e + 1) % config['SAVE_MODEL'] == 0:
            print('save model')
            save_model(e+1)

    print()
    print('Finish training | exec time: %.4f [sec]' % (time.time() - start))


def validation(XU_valid, YU_valid, XD_valid, YD_valid, model, utt_vocab, config):
    model.eval()
    indexes = [i for i in range(len(XU_valid))]
    random.shuffle(indexes)
    criterion = nn.CrossEntropyLoss()
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
        (Xordered, Xmisordered, Xtarget), (DAordered, DAmisordered, DAtarget), Y = make_triple(utterance_pairs, utt_vocab, da_pairs)
        loss, preds = model.forward(XOrdered=Xordered, XMisOrdered=Xmisordered, XTarget=Xtarget,
                                    DAOrdered=DAordered, DAMisOrdered=DAmisordered, DATarget=DAtarget,
                                    Y=Y, step_size=step_size, criterion=criterion)
        result = [0 if line[0] > line[1] else 1 for line in preds]
        valid_acc.append(accuracy_score(y_true=Y.data.tolist(), y_pred=result))
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
                                              utterance_hidden=config['SSN_ENC_HIDDEN'], padding_idx=utt_vocab.word2id['<PAD>']).to(device)
    utterance_pair_encoder.load_state_dict(torch.load(os.path.join(config['log_dir'], 'utt_pair_enc_statevalidbest.model')))
    if config['use_da']:
        da_pair_encoder = DAPairEncoder(da_hidden_size=config['SSN_DA_HIDDEN'], da_embed_size=config['SSN_DA_EMBED'], da_vocab_size=len(da_vocab.word2id)).to(device)
        da_pair_encoder.load_state_dict(torch.load(os.path.join(config['log_dir'], 'da_pair_enc_')))
    else:
        da_pair_encoder = None
    order_reasoning_layer = OrderReasoningLayer(encoder_hidden_size=config['SSN_ENC_HIDDEN'], hidden_size=config['SSN_REASONING_HIDDEN'],
                                                middle_layer_size=config['SSN_MIDDLE_LAYER'], da_hidden_size=config['SSN_DA_HIDDEN'], config=config).to(device)
    predictor = OrderPredictor(utterance_pair_encoder=utterance_pair_encoder, order_reasoning_layer=order_reasoning_layer,
                               da_pair_encoder=da_pair_encoder, config=config, device=device).to(device)
    criterion = nn.CrossEntropyLoss()
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
            (Xordered, Xmisordered, Xtarget), (DAordered, DAmisordered, DAtarget), Y = make_triple(utterance_pairs, utt_vocab, da_pairs)
            loss, preds = predictor.forward(XOrdered=Xordered, XMisOrdered=Xmisordered, XTarget=Xtarget,
                                        DAOrdered=DAordered, DAMisOrdered=DAmisordered, DATarget=DAtarget,
                                        Y=Y, step_size=step_size, criterion=criterion)
            result = [0 if line[0] > line[1] else 1 for line in preds]
            y_preds.extend(result)
            y_trues.extend(Y.data.tolist())
            k += step_size
        acc.append(accuracy_score(y_pred=y_preds, y_true=y_trues))
    print('Avg. of Accuracy: {}'.format(np.mean(acc)))

def make_triple(utterance_pairs, utt_vocab, da_pairs=None):
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
        (ordered, misordered, target), (da_ordered, da_misordered, da_target), label = sample_triple(utterance_pairs[bidx], da_seq)
        Xordered.append(ordered)
        Xmisordered.append(misordered)
        Xtarget.append(target)
        DAordered.append(da_ordered)
        DAmisordered.append(da_misordered)
        DAtarget.append(da_target)
        Y.append(label)
    # padding
    Xordered = padding(Xordered, utt_vocab.word2id['<PAD>'])
    Xmisordered = padding(Xmisordered, utt_vocab.word2id['<PAD>'])
    Xtarget = padding(Xtarget, utt_vocab.word2id['<PAD>'])
    if not da_pairs is None:
        da_ordered = torch.tensor(DAordered).to(device)
        da_misordered = torch.tensor(DAmisordered).to(device)
        da_target = torch.tensor(DAtarget).to(device)
    Y = torch.tensor(Y).to(device)
    return (Xordered, Xmisordered, Xtarget), (da_ordered, da_misordered, da_target), Y


def sample_triple(pairs, da_pairs):
    sampled_idx = random.sample([i for i in range(len(pairs)-1)], 3)
    i, j, k = sorted(sampled_idx)
    ordered = [pairs[i], pairs[j], pairs[k]]
    misordered = [pairs[i], pairs[k], pairs[j]]
    if da_pairs is None:
        da_ordered = None
        da_misordered = None
    else:
        da_ordered = [da_pairs[i], da_pairs[j], da_pairs[k]]
        da_misordered = [da_pairs[i], da_pairs[k], da_pairs[j]]
    if random.random() > 0.5:
        target = [pairs[i], pairs[j], pairs[-1]]
        if da_pairs is None:
            da_target = None
        else:
            da_target = [da_pairs[i], da_pairs[j], da_pairs[-1]]
        label = [0]
    else:
        target = [pairs[i], pairs[-1], pairs[j]]
        if da_pairs is None:
            da_target = None
        else:
            da_target = [da_pairs[i], da_pairs[-1], da_pairs[j]]
        label = [1]
    return (ordered, misordered, target), (da_ordered, da_misordered, da_target), label


def padding(batch, pad_idx):
    pair_list = []
    for i in range(len(batch[0])):
        max_seq_len = max(len(b[i]) + 1 for b in batch)
        for ci in range(len(batch)):
            batch[ci][i] = batch[ci][i] + [pad_idx] * (max_seq_len - len(batch[ci][i]))
        pair_list.append(torch.tensor([b[i] for b in batch]).to(device))
    return pair_list


if __name__ == '__main__':
    global args, device
    args, device = parse()
    train(args.expr)