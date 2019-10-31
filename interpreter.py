import time
import os
import pyhocon
import torch
from torch import optim
from models import *
from nn_blocks import *
from utils import *
from train import initialize_env, create_DAdata, create_Uttdata, parse
import argparse
from pprint import pprint
import numpy as np
import pickle
import random


def interpreter(experiment):
    print('load vocab')
    config = initialize_env(experiment)
    # X_train, Y_train, X_valid, Y_valid, X_test, Y_test, _, _, turn = create_DAdata(config)
    # da_vocab = da_Vocab(config, X_train + X_valid, Y_train + Y_valid)
    # XU_train, YU_train, XU_valid, YU_valid, XU_test, YU_test = create_Uttdata(config)
    # utt_vocab = utt_Vocab(config, XU_train + XU_valid, YU_train + YU_valid)
    da_vocab = da_Vocab(config=config, create_vocab=False)
    utt_vocab = utt_Vocab(config=config, create_vocab=False)

    print('load models')
    encoder, context, decoder = None, None, None
    if config['use_da']:
        encoder = DAEncoder(da_input_size=len(da_vocab.word2id), da_embed_size=config['DA_EMBED'],
                        da_hidden=config['DA_HIDDEN']).to(device)
        context = DAContextEncoder(da_hidden=config['DA_HIDDEN']).to(device)
        encoder.load_state_dict(torch.load(os.path.join(config['log_dir'], 'enc_state{}.model'.format(args.epoch))))
        context.load_state_dict(torch.load(os.path.join(config['log_dir'], 'context_state{}.model'.format(args.epoch))))

        decoder = DADecoder(da_input_size=len(da_vocab.word2id), da_embed_size=config['DA_EMBED'],
                            da_hidden=config['DEC_HIDDEN']).to(device)

        decoder.load_state_dict(torch.load(os.path.join(config['log_dir'], 'dec_state{}.model'.format(args.epoch))))

    utt_encoder = UtteranceEncoder(utt_input_size=len(utt_vocab.word2id), embed_size=config['UTT_EMBED'], utterance_hidden=config['UTT_HIDDEN'], padding_idx=utt_vocab.word2id['<UttPAD>'], fine_tuning=True).to(device)
    utt_decoder = UtteranceDecoder(utterance_hidden_size=config['DEC_HIDDEN'], utt_embed_size=config['UTT_EMBED'], utt_vocab_size=config['UTT_MAX_VOCAB']).to(device)
    utt_encoder.load_state_dict(torch.load(os.path.join(config['log_dir'], 'utt_enc_state{}.model'.format(args.epoch))))
    utt_decoder.load_state_dict(torch.load(os.path.join(config['log_dir'], 'utt_dec_state{}.model'.format(args.epoch))))

    utt_context = UtteranceContextEncoder(utterance_hidden_size=config['UTT_CONTEXT']).to(device)
    utt_context.load_state_dict(torch.load(os.path.join(config['log_dir'], 'utt_context_state{}.model'.format(args.epoch))))

    model = EncoderDecoderModel(da_vocab=da_vocab, utt_vocab=utt_vocab, device=device,
                                da_encoder=encoder, utt_encoder=utt_encoder, da_context=context,
                                utt_context=utt_context, da_decoder=decoder, utt_decoder=utt_decoder, config=config).to(device)

    da_context_hidden = context.initHidden(1, device) if config['use_da'] else None
    utt_context_hidden = utt_context.initHidden(1, device) if config['use_uttcontext'] else None

    print('ok, i\'m ready.')

    while 1:

        utterance = input('>> ').lower()

        if utterance == 'exit' or utterance == 'bye':
            print('see you again.')
            break

        print('Select Dialogue Act (Statement, Understanding, Greeting, Agreement, Apology, Directive, Question, Uninterpretable, Other)')
        input_da = '<{}>'.format(input('>>'))

        XU_seq = en_preprocess(utterance)
        XU_seq = ['<BOS>'] + XU_seq + ['<EOS>']
        XU_seq = [utt_vocab.word2id[word] if word in utt_vocab.word2id.keys() else utt_vocab.word2id['<UNK>'] for word in XU_seq]

        # TODO: How to deal utterance's DA
        DA = da_vocab.word2id[input_da]
        X_tensor = torch.tensor([[DA]]).to(device)

        XU_tensor = torch.tensor([XU_seq]).to(device)


        pred_seq, da_context_hidden, utt_context_hidden, decoder_output = model.predict(X_da=X_tensor, X_utt=XU_tensor,
                                                                      da_context_hidden=da_context_hidden, utt_context_hidden=utt_context_hidden)
        pred_idx = torch.argmax(decoder_output) if config['use_da'] else 0


        print('{} ({})'.format(' '.join([utt_vocab.id2word[wid] for wid in pred_seq]), da_vocab.id2word[pred_idx.item()]))

        print()

    return 0

if __name__ == '__main__':
    global args, device
    args, device = parse()
    interpreter(args.expr)

