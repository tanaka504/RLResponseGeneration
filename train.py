import time
from torch import optim
from models import *
from utils import *
from nn_blocks import *
import random
from NLI import NLI
from order_predict import OrderPredictor


def train(experiment, fine_tuning=False):
    print('loading setting "{}"...'.format(experiment))
    config = initialize_env(experiment)
    X_train, Y_train, XU_train, YU_train = create_traindata(config=config, prefix='train')
    X_valid, Y_valid, XU_valid, YU_valid = create_traindata(config=config, prefix='valid')
    print('Finish create train data...')

    if os.path.exists(os.path.join(config['log_root'], 'da_vocab.dict')):
        da_vocab = da_Vocab(config, create_vocab=False)
        utt_vocab = utt_Vocab(config, create_vocab=False)
    else:
        da_vocab = da_Vocab(config, das=[token for conv in X_train + X_valid + Y_train + Y_valid for token in conv])
        utt_vocab = utt_Vocab(config, sentences=[sentence for conv in XU_train + XU_valid + YU_train + YU_valid for sentence in conv])
        da_vocab.save()
        utt_vocab.save()
    print('Finish create vocab dic...')

    # Tokenize sequences
    X_train, Y_train = da_vocab.tokenize(X_train), da_vocab.tokenize(Y_train)
    X_valid, Y_valid = da_vocab.tokenize(X_valid), da_vocab.tokenize(Y_valid)
    XU_train, YU_train = utt_vocab.tokenize(XU_train), utt_vocab.tokenize(YU_train)
    XU_valid, YU_valid = utt_vocab.tokenize(XU_valid), utt_vocab.tokenize(YU_valid)
    print('Finish preparing dataset...')
    assert len(X_train) == len(Y_train), 'Unexpect content in train data'
    assert len(X_valid) == len(Y_valid), 'Unexpect content in valid data'

    lr = config['lr']
    batch_size = config['BATCH_SIZE']
    plot_losses = []
    print_total_loss = 0
    plot_total_loss = 0

    if 'HRED' in args.expr:
        pretrain_model = experiment + '_pretrain'
    else:
        pretrain_model = 'HRED'

    # construct models
    utt_encoder = UtteranceEncoder(utt_input_size=len(utt_vocab.word2id), embed_size=config['UTT_EMBED'], utterance_hidden=config['UTT_HIDDEN'], padding_idx=utt_vocab.word2id['<PAD>'], fine_tuning=False).cuda()
    utt_decoder = UtteranceDecoder(utterance_hidden_size=config['DEC_HIDDEN'], utt_embed_size=config['UTT_EMBED'], utt_vocab_size=len(utt_vocab.word2id)).cuda()
    # requires_grad が True のパラメータのみをオプティマイザにのせる
    utt_encoder_opt = optim.Adam(list(filter(lambda x: x.requires_grad, utt_encoder.parameters())), lr=lr)
    utt_decoder_opt = optim.Adam(utt_decoder.parameters(), lr=lr)
    utt_context = UtteranceContextEncoder(utterance_hidden_size=config['UTT_CONTEXT']).cuda()
    utt_context_opt = optim.Adam(utt_context.parameters(), lr=lr)

    if fine_tuning:
        print('fine tuning')
        utt_encoder.load_state_dict(torch.load(os.path.join(config['log_root'], pretrain_model, 'utt_enc_state{}.model'.format(args.epoch)), map_location=lambda storage, loc: storage))
        utt_decoder.load_state_dict(torch.load(os.path.join(config['log_root'], pretrain_model, 'utt_dec_state{}.model'.format(args.epoch)), map_location=lambda storage, loc: storage))
        utt_context.load_state_dict(torch.load(os.path.join(config['log_root'], pretrain_model, 'utt_context_state{}.model'.format(args.epoch)), map_location=lambda storage, loc: storage))

    if 'HRED' in args.expr:

        model = HRED(utt_vocab=utt_vocab,
                     utt_encoder=utt_encoder, utt_context=utt_context,
                     utt_decoder=utt_decoder, config=config).cuda()
    else:
        ord_predictor = OrderPredictor(utt_vocab=utt_vocab, da_vocab=da_vocab, config=config).cuda()
        ord_predictor.load_state_dict(torch.load(os.path.join(config['log_root'], 'order_predict', 'orderpred_statevalidbest.model')))
        nli_model = NLI().cuda()
        model = RL(utt_vocab=utt_vocab,
                   utt_encoder=utt_encoder,
                   utt_context=utt_context,
                   utt_decoder=utt_decoder,
                   nli_model=nli_model, ssn_model=ord_predictor,
                   config=config).cuda()
    print('Success construct model...')
    criterion = nn.CrossEntropyLoss(ignore_index=utt_vocab.word2id['<PAD>'], reduce=False)
    print('---start training---')

    start = time.time()
    _valid_loss = None
    _train_loss = None
    early_stop = 0
    for e in range(config['EPOCH']):
        tmp_time = time.time()
        print('Epoch {} start'.format(e+1))

        indexes = [i for i in range(len(X_train))]
        random.shuffle(indexes)
        k = 0
        model.train()
        while k < len(indexes):
            # initialize
            step_size = min(batch_size, len(indexes) - k)
            batch_idx = indexes[k : k + step_size]
            utt_context_hidden = utt_context.initHidden(step_size)
            utt_encoder_opt.zero_grad()
            utt_decoder_opt.zero_grad()
            utt_context_opt.zero_grad()

            # create batch data
            print('\rConversation {}/{} training...'.format(k + step_size, len(X_train)), end='')

            XU_seq = [XU_train[seq_idx] for seq_idx in batch_idx]
            YU_seq = [YU_train[seq_idx] for seq_idx in batch_idx]
            max_conv_len = max(len(s) for s in XU_seq)  # seq_len は DA と UTT で共通

            for i in range(0, max_conv_len):
                max_xseq_len = max(len(XU[i]) + 1 for XU in XU_seq)
                max_yseq_len = max(len(YU[i]) + 1 for YU in YU_seq)

                # utterance padding
                for ci in range(len(XU_seq)):
                    XU_seq[ci][i] = XU_seq[ci][i] + [utt_vocab.word2id['<PAD>']] * (max_xseq_len - len(XU_seq[ci][i]))
                    YU_seq[ci][i] = YU_seq[ci][i] + [utt_vocab.word2id['<PAD>']] * (max_yseq_len - len(YU_seq[ci][i]))
                XU_tensor = torch.tensor([XU[i] for XU in XU_seq]).cuda()
                YU_tensor = torch.tensor([YU[i] for YU in YU_seq]).cuda()

                # XU_tensor = (batch_size, seq_len)
                last = True if i == max_conv_len - 1 else False
                if last:
                    loss, utt_context_hidden, _ = model.forward(X_utt=XU_tensor, Y_utt=YU_tensor, context=XU_seq,
                                                                step_size=step_size,
                                                                utt_context_hidden=utt_context_hidden,
                                                                criterion=criterion, last=last)
                    print_total_loss += loss
                    plot_total_loss += loss
                    utt_encoder_opt.step()
                    utt_decoder_opt.step()
                    utt_context_opt.step()
                else:
                    loss, utt_context_hidden, _ = model.forward(X_utt=XU_tensor, Y_utt=YU_tensor, step_size=step_size,
                                                   utt_context_hidden=utt_context_hidden,
                                                   criterion=criterion, last=last)
            k += step_size

        print()
        valid_loss, valid_reward = validation(XU_valid=XU_valid, YU_valid=YU_valid, model=model, utt_context=utt_context, utt_vocab=utt_vocab, config=config)

        def save_model(filename):
            torch.save(utt_encoder.state_dict(), os.path.join(config['log_dir'], 'utt_enc_state{}.model'.format(filename)))
            torch.save(utt_decoder.state_dict(), os.path.join(config['log_dir'], 'utt_dec_state{}.model'.format(filename)))
            torch.save(utt_context.state_dict(), os.path.join(config['log_dir'], 'utt_context_state{}.model'.format(filename)))

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
                print('early stopping count | {}/{}'.format(early_stop, config['EARLY_STOP']))
                if early_stop >= config['EARLY_STOP']:
                    break
        if (e + 1) % config['LOGGING_FREQ'] == 0:
            print_loss_avg = print_total_loss / config['LOGGING_FREQ']
            print_total_loss = 0
            print('steps %d\tloss %.4f\tvalid loss %.4f\tvalid reward %.4f | exec time %.4f' % (e + 1, print_loss_avg, valid_loss, valid_reward, time.time() - tmp_time))
            plot_loss_avg = plot_total_loss / config['LOGGING_FREQ']
            plot_losses.append(plot_loss_avg)
            plot_total_loss = 0

        if (e + 1) % config['SAVE_MODEL'] == 0:
            print('saving model')
            save_model(e+1)

    print()
    print('Finish training | exec time: %.4f [sec]' % (time.time() - start))


def validation(XU_valid, YU_valid, model, utt_context, utt_vocab, config):
    model.eval()

    criterion = nn.CrossEntropyLoss(ignore_index=utt_vocab.word2id['<PAD>'], reduce=False)
    total_loss = 0
    k = 0
    batch_size = config['BATCH_SIZE']
    indexes = [i for i in range(len(XU_valid))]
    random.shuffle(indexes)

    while k < len(indexes):
        step_size = min(batch_size, len(indexes) - k)
        batch_idx = indexes[k: k + step_size]
        XU_seq = [XU_valid[seq_idx] for seq_idx in batch_idx]
        YU_seq = [YU_valid[seq_idx] for seq_idx in batch_idx]
        utt_context_hidden = utt_context.initHidden(step_size)
        max_conv_len = max(len(s) for s in XU_seq)
        for i in range(0, max_conv_len):
            max_xseq_len = max(len(XU[i]) + 1 for XU in XU_seq)
            max_yseq_len = max(len(YU[i]) + 1 for YU in YU_seq)
            for ci in range(len(XU_seq)):
                XU_seq[ci][i] = XU_seq[ci][i] + [utt_vocab.word2id['<PAD>']] * (max_xseq_len - len(XU_seq[ci][i]))
                YU_seq[ci][i] = YU_seq[ci][i] + [utt_vocab.word2id['<PAD>']] * (max_yseq_len - len(YU_seq[ci][i]))
            XU_tensor = torch.tensor([x[i] for x in XU_seq]).cuda()
            YU_tensor = torch.tensor([y[i] for y in YU_seq]).cuda()
            loss, utt_context_hidden, reward, pred_seq = model.forward(X_utt=XU_tensor, Y_utt=YU_tensor,
                                                  utt_context_hidden=utt_context_hidden, criterion=criterion, step_size=step_size, last=False)
            total_loss += loss
        k += step_size
    sample_idx = random.sample([i for i in range(len(XU_seq))], 3)
    for idx in sample_idx:
        context = ' '.join([utt_vocab.id2word[wid] for wid in XU_seq[idx][-1]])
        hyp = ' '.join([utt_vocab.id2word[wid] for wid in pred_seq[idx]])
        print('context:\t{}'.format(context))
        print('hyp:\t{}'.format(hyp))
    return total_loss, reward

if __name__ == '__main__':
    args = parse()
    fine_tuning = False if 'pretrain' in args.expr else True
    train(args.expr, fine_tuning=fine_tuning)
