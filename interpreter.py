import time
import os
import pyhocon
import torch
from torch import optim
from models import *
from nn_blocks import *
from utils import *
from pprint import pprint
import numpy as np
import pickle
import random


def interpreter(experiment):
    print('load vocab')
    config = initialize_env(experiment)
    da_vocab = da_Vocab(config=config, create_vocab=False)
    utt_vocab = utt_Vocab(config=config, create_vocab=False)

    print('load models')
    utt_encoder = UtteranceEncoder(utt_input_size=len(utt_vocab.word2id), embed_size=config['UTT_EMBED'], utterance_hidden=config['UTT_HIDDEN'], padding_idx=utt_vocab.word2id['<UttPAD>'], fine_tuning=True).to(device)
    utt_decoder = UtteranceDecoder(utterance_hidden_size=config['DEC_HIDDEN'], utt_embed_size=config['UTT_EMBED'], utt_vocab_size=config['UTT_MAX_VOCAB']).to(device)
    utt_encoder.load_state_dict(torch.load(os.path.join(config['log_dir'], 'utt_enc_state{}.model'.format(args.epoch))))
    utt_decoder.load_state_dict(torch.load(os.path.join(config['log_dir'], 'utt_dec_state{}.model'.format(args.epoch))))

    utt_context = UtteranceContextEncoder(utterance_hidden_size=config['UTT_CONTEXT']).to(device)
    utt_context.load_state_dict(torch.load(os.path.join(config['log_dir'], 'utt_context_state{}.model'.format(args.epoch))))

    model = RL(utt_vocab=utt_vocab,
                utt_encoder=utt_encoder, utt_context=utt_context, utt_decoder=utt_decoder, config=config).to(device)

    utt_context_hidden = utt_context.initHidden(1) if config['use_uttcontext'] else None

    print('ok, i\'m ready.')

    while 1:
        utterance = input('>> ').lower()
        if utterance == 'exit' or utterance == 'bye':
            print('see you again.')
            break
        XU_seq = en_preprocess(utterance)
        XU_seq = ['<BOS>'] + XU_seq + ['<EOS>']
        XU_seq = [utt_vocab.word2id[word] if word in utt_vocab.word2id.keys() else utt_vocab.word2id['<UNK>'] for word in XU_seq]
        # X_tensor = torch.tensor([[DA]]).to(device)
        XU_tensor = torch.tensor([XU_seq]).cuda()
        pred_seq, utt_context_hidden = model.predict(X_utt=XU_tensor, utt_context_hidden=utt_context_hidden)
        print(' '.join([utt_vocab.id2word[wid] for wid in pred_seq]))
        print()

    return 0

if __name__ == '__main__':
    args = parse()
    interpreter(args.expr)

