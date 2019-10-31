import time
import os
import pyhocon
import pickle
import torch
from torch import nn
from torch import optim
from models import *
from utils import *
from nn_blocks import *
import argparse
import random


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--expr', '-e', default='DAonly', help='input experiment config')
    parser.add_argument('--gpu', '-g', type=int, default=0, help='input gpu num')
    parser.add_argument('--epoch', default=10)
    args = parser.parse_args()
    
    if torch.cuda.is_available():
        device = torch.device('cuda:{}'.format(args.gpu))
    else:
        device = 'cpu'

    print('Use device: ', device)

    return args, device


def initialize_env(name):
    config = pyhocon.ConfigFactory.parse_file('experiments.conf')[name]
    config['log_dir'] = os.path.join(config['log_root'], name)
    if not os.path.exists(config['log_dir']):
        os.makedirs(config['log_dir'])

    return config


def make_batchidx(X):
    length = {}
    for idx, conv in enumerate(X):
        if len(conv) in length:
            length[len(conv)].append(idx)
        else:
            length[len(conv)] = [idx]
    return [v for k, v in sorted(length.items(), key=lambda x: x[0])]


def train(experiment, fine_tuning=False):
    print('loading setting "{}"...'.format(experiment))

    config = initialize_env(experiment)
    X_train, Y_train, XU_train, YU_train = create_traindata(config=config, prefix='train')
    X_valid, Y_valid, XU_valid, YU_valid = create_traindata(config=config, prefix='dev')
    print('Finish create train data...')

    if os.path.exists(os.path.join(config['log_root'], 'da_vocab.dict')):
        da_vocab = da_Vocab(config, create_vocab=False)
        utt_vocab = utt_Vocab(config, create_vocab=False)
    else:
        da_vocab = da_Vocab(config, X_train + X_valid, Y_train + Y_valid)
        utt_vocab = utt_Vocab(config, XU_train + XU_valid, YU_train + YU_valid)
        da_vocab.save()
        utt_vocab.save()

    print('Finish create vocab dic...')


    # Tokenize sequences
    X_train, Y_train = da_vocab.tokenize(X_train, Y_train)
    X_valid, Y_valid = da_vocab.tokenize(X_valid, Y_valid)

    XU_train, YU_train = utt_vocab.tokenize(XU_train, YU_train)
    XU_valid, YU_valid = utt_vocab.tokenize(XU_valid, YU_valid)
    print('Finish preparing dataset...')

    assert len(X_train) == len(Y_train), 'Unexpect content in train data'
    assert len(X_valid) == len(Y_valid), 'Unexpect content in valid data'

    lr = config['lr']
    batch_size = config['BATCH_SIZE']
    plot_losses = []

    print_total_loss = 0
    plot_total_loss = 0

    pretrain_model = experiment + '_pretrain'

    # constract models
    if config['use_da']:
        da_vocab_size = len(da_vocab.word2id)
        da_encoder = DAEncoder(da_input_size=da_vocab_size, da_embed_size=config['DA_EMBED'], da_hidden=config['DA_HIDDEN']).to(device)
        da_context = DAContextEncoder(da_hidden=config['DA_HIDDEN']).to(device)
        da_decoder = DADecoder(da_input_size=len(da_vocab.word2id), da_embed_size=config['DA_EMBED'],
                               da_hidden=config['DEC_HIDDEN']).to(device)
        da_encoder_opt = optim.Adam(da_encoder.parameters(), lr=lr)
        da_context_opt = optim.Adam(da_context.parameters(), lr=lr)
        da_decoder_opt = optim.Adam(da_decoder.parameters(), lr=lr)

        if fine_tuning:
            da_encoder.load_state_dict(torch.load(os.path.join(config['log_root'], pretrain_model, 'enc_state{}.model'.format(args.epoch))))
            da_context.load_state_dict(torch.load(os.path.join(config['log_root'], pretrain_model, 'context_state{}.model'.format(args.epoch))))
    else:
        da_encoder = None
        da_context = None
        da_decoder = None

    utt_encoder = UtteranceEncoder(utt_input_size=len(utt_vocab.word2id), embed_size=config['UTT_EMBED'], utterance_hidden=config['UTT_HIDDEN'], padding_idx=utt_vocab.word2id['<UttPAD>'], fine_tuning=fine_tuning).to(device)
    utt_decoder = UtteranceDecoder(utterance_hidden_size=config['DEC_HIDDEN'], utt_embed_size=config['UTT_EMBED'], utt_vocab_size=len(utt_vocab.word2id)).to(device)
    # requires_grad が True のパラメータのみをオプティマイザにのせる
    utt_encoder_opt = optim.Adam(list(filter(lambda x: x.requires_grad, utt_encoder.parameters())), lr=lr)
    utt_decoder_opt = optim.Adam(utt_decoder.parameters(), lr=lr)

    utt_context = UtteranceContextEncoder(utterance_hidden_size=config['UTT_CONTEXT']).to(device)
    utt_context_opt = optim.Adam(utt_context.parameters(), lr=lr)

    if fine_tuning:
        print('fine tuning')
        utt_encoder.load_state_dict(
            torch.load(os.path.join(config['log_root'], pretrain_model, 'utt_enc_state{}.model'.format(args.epoch))))
        utt_decoder.load_state_dict(
            torch.load(os.path.join(config['log_root'], pretrain_model, 'utt_dec_state{}.model'.format(args.epoch))))
        utt_context.load_state_dict(torch.load(os.path.join(config['log_root'], pretrain_model, 'utt_context_state{}.model'.format(args.epoch))))

    model = EncoderDecoderModel(da_vocab=da_vocab, utt_vocab=utt_vocab, device=device,
                                da_encoder=da_encoder, utt_encoder=utt_encoder,
                                da_context=da_context, utt_context=utt_context,
                                da_decoder=da_decoder, utt_decoder=utt_decoder, config=config).to(device)
    print('Success construct model...')


    criterion = nn.CrossEntropyLoss(ignore_index=utt_vocab.word2id['<UttPAD>'], reduce=False)
    da_criterion = nn.CrossEntropyLoss(ignore_index=da_vocab.word2id['<PAD>'], reduce=False)

    print('---start training---')

    start = time.time()
    _valid_loss = None

    for e in range(config['EPOCH']):
        tmp_time = time.time()
        print('Epoch {} start'.format(e+1))

        indexes = [i for i in range(len(X_train))]
        random.shuffle(indexes)
        k = 0
        while k < len(indexes):
            # initialize
            step_size = min(batch_size, len(indexes) - k)

            context_hidden = da_context.initHidden(step_size, device) if config['use_da'] else None
            utt_context_hidden = utt_context.initHidden(step_size, device)

            if config['use_da']:
                da_context_opt.zero_grad()
                da_encoder_opt.zero_grad()
                da_decoder_opt.zero_grad()

            utt_encoder_opt.zero_grad()
            utt_decoder_opt.zero_grad()
            utt_context_opt.zero_grad()

            batch_idx = indexes[k : k + step_size]

            # create batch data
            print('\rConversation {}/{} training...'.format(k + step_size, len(X_train)), end='')

            X_seq = [X_train[seq_idx] for seq_idx in batch_idx]
            Y_seq = [Y_train[seq_idx] for seq_idx in batch_idx]
            turn_seq = [Tturn[seq_idx] for seq_idx in batch_idx]
            max_conv_len = max(len(s) for s in X_seq)  # seq_len は DA と UTT で共通

            XU_seq = [XU_train[seq_idx] for seq_idx in batch_idx]
            YU_seq = [YU_train[seq_idx] for seq_idx in batch_idx]


            assert len(X_seq) == len(Y_seq), 'Unexpect sequence length'

            for i in range(0, max_conv_len):
                X_tensor = torch.tensor([[X[i]] for X in X_seq]).to(device)
                Y_tensor = torch.tensor([[Y[i]] for Y in Y_seq]).to(device)

                turn_tensor = torch.tensor([[t[i]] for t in turn_seq]).to(device)

                max_xseq_len = max(len(XU[i]) + 1 for XU in XU_seq)
                max_yseq_len = max(len(YU[i]) + 1 for YU in YU_seq)

                # utterance padding
                for ci in range(len(XU_seq)):
                    XU_seq[ci][i] = XU_seq[ci][i] + [utt_vocab.word2id['<UttPAD>']] * (max_xseq_len - len(XU_seq[ci][i]))
                    YU_seq[ci][i] = YU_seq[ci][i] + [utt_vocab.word2id['<UttPAD>']] * (max_yseq_len - len(YU_seq[ci][i]))
                XU_tensor = torch.tensor([XU[i] for XU in XU_seq]).to(device)
                YU_tensor = torch.tensor([YU[i] for YU in YU_seq]).to(device)

                # X_tensor = (batch_size, 1)
                # XU_tensor = (batch_size, seq_len)

                last = True if i == max_conv_len - 1 else False
    
                if last:
                    loss, context_hidden, utt_context_hidden = model.forward(X_da=X_tensor, Y_da=Y_tensor, X_utt=XU_tensor, Y_utt=YU_tensor, step_size=step_size,
                                                         da_context_hidden=context_hidden, utt_context_hidden=utt_context_hidden,
                                                         criterion=criterion, da_criterion=da_criterion, last=last)
                    print_total_loss += loss
                    plot_total_loss += loss

                    if config['use_da']:
                        da_encoder_opt.step()
                        da_context_opt.step()
                        da_decoder_opt.step()
                    utt_encoder_opt.step()
                    utt_decoder_opt.step()
                    utt_context_opt.step()

                else:
                    loss, context_hidden, utt_context_hidden = model.forward(X_da=X_tensor, Y_da=Y_tensor, X_utt=XU_tensor, Y_utt=YU_tensor, step_size=step_size,
                                                   da_context_hidden=context_hidden, utt_context_hidden=utt_context_hidden,
                                                   criterion=criterion, da_criterion=da_criterion, last=last)


            k += step_size

        print()
        valid_loss = validation(X_valid=X_valid, Y_valid=Y_valid, XU_valid=XU_valid, YU_valid=YU_valid, turn=Vturn,
                                model=model, da_encoder=da_encoder, da_decoder=da_decoder, da_context=da_context, da_vocab=da_vocab,
                                utt_encoder=utt_encoder, utt_context=utt_context, utt_decoder=utt_decoder, utt_vocab=utt_vocab, config=config)

        if _valid_loss is None:
            if config['use_da']:
                torch.save(da_encoder.state_dict(), os.path.join(config['log_dir'], 'enc_statebest.model'))
                torch.save(da_context.state_dict(), os.path.join(config['log_dir'], 'context_statebest.model'))
                torch.save(da_decoder.state_dict(), os.path.join(config['log_dir'], 'dec_statebest.model'))

            torch.save(utt_encoder.state_dict(), os.path.join(config['log_dir'], 'utt_enc_statebest.model'))
            torch.save(utt_decoder.state_dict(), os.path.join(config['log_dir'], 'utt_dec_statebest.model'))
            torch.save(utt_context.state_dict(), os.path.join(config['log_dir'], 'utt_context_statebest.model'))

            _valid_loss = valid_loss

        else:
            if _valid_loss > valid_loss:
                if config['use_da']:
                    torch.save(da_encoder.state_dict(), os.path.join(config['log_dir'], 'enc_statebest.model'))
                    torch.save(da_context.state_dict(), os.path.join(config['log_dir'], 'context_statebest.model'))
                    torch.save(da_decoder.state_dict(), os.path.join(config['log_dir'], 'dec_statebest.model'))

                torch.save(utt_encoder.state_dict(), os.path.join(config['log_dir'], 'utt_enc_statebest.model'))
                torch.save(utt_decoder.state_dict(), os.path.join(config['log_dir'], 'utt_dec_statebest.model'))
                torch.save(utt_context.state_dict(), os.path.join(config['log_dir'], 'utt_context_statebest.model'))

                _valid_loss = valid_loss


        if (e + 1) % config['LOGGING_FREQ'] == 0:
            print_loss_avg = print_total_loss / config['LOGGING_FREQ']
            print_total_loss = 0
            print('steps %d\tloss %.4f\tvalid loss %.4f | exec time %.4f' % (e + 1, print_loss_avg, valid_loss, time.time() - tmp_time))
            plot_loss_avg = plot_total_loss / config['LOGGING_FREQ']
            plot_losses.append(plot_loss_avg)
            plot_total_loss = 0

        if (e + 1) % config['SAVE_MODEL'] == 0:
            print('saving model')
            if config['use_da']:
                torch.save(da_encoder.state_dict(), os.path.join(config['log_dir'], 'enc_state{}.model'.format(e + 1)))
                torch.save(da_context.state_dict(), os.path.join(config['log_dir'], 'context_state{}.model'.format(e + 1)))
                torch.save(da_decoder.state_dict(), os.path.join(config['log_dir'], 'dec_state{}.model'.format(e + 1)))

            torch.save(utt_encoder.state_dict(), os.path.join(config['log_dir'], 'utt_enc_state{}.model'.format(e + 1)))
            torch.save(utt_decoder.state_dict(), os.path.join(config['log_dir'], 'utt_dec_state{}.model'.format(e + 1)))
            torch.save(utt_context.state_dict(), os.path.join(config['log_dir'], 'utt_context_state{}.model'.format(e + 1)))

    print()
    print('Finish training | exec time: %.4f [sec]' % (time.time() - start))

def validation(X_valid, Y_valid, XU_valid, YU_valid, model, turn,
               da_encoder, da_decoder, da_context, da_vocab,
               utt_encoder, utt_context, utt_decoder, utt_vocab, config):

    da_context_hidden = da_context.initHidden(1, device) if config['use_da'] else None
    utt_context_hidden = utt_context.initHidden(1, device) if config['use_uttcontext'] else None
    criterion = nn.CrossEntropyLoss(ignore_index=utt_vocab.word2id['<UttPAD>'], reduce=False)
    da_criterion = nn.CrossEntropyLoss(ignore_index=da_vocab.word2id['<PAD>'], reduce=False)
    total_loss = 0

    for seq_idx in range(len(X_valid)):
        X_seq = X_valid[seq_idx]
        Y_seq = Y_valid[seq_idx]
        XU_seq = XU_valid[seq_idx]
        YU_seq = YU_valid[seq_idx]

        assert len(X_seq) == len(Y_seq), 'Unexpect sequence len in evaluate {} != {}'.format(len(X_seq), len(Y_seq))

        for i in range(0, len(X_seq)):
            X_tensor = torch.tensor([[X_seq[i]]]).to(device)
            Y_tensor = torch.tensor([[Y_seq[i]]]).to(device)

            XU_tensor = torch.tensor([XU_seq[i]]).to(device)
            YU_tensor = torch.tensor([YU_seq[i]]).to(device)

            loss, context_hidden, utt_context_hidden = model.evaluate(X_da=X_tensor, Y_da=Y_tensor, X_utt=XU_tensor, Y_utt=YU_tensor,
                                                  da_context_hidden=da_context_hidden, utt_context_hidden=utt_context_hidden,
                                                  criterion=criterion, da_criterion=da_criterion)

            total_loss += loss
    return total_loss

if __name__ == '__main__':
    global args, device
    args, device = parse()
    fine_tuning = False if 'pretrain' in args.expr else True
    train(args.expr, fine_tuning=fine_tuning)
