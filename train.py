import time
from torch import optim
from models import *
from utils import *
from nn_blocks import *
import random
from NLI import NLI
from order_predict import OrderPredictor


class Reward:
    def __init__(self, utt_vocab, da_vocab, config):
        self.ssn_model = OrderPredictor(utt_vocab=utt_vocab, da_vocab=da_vocab, config=config).cuda()
        self.ssn_model.load_state_dict(torch.load(os.path.join(config['log_dir'], 'orderpred_statevalidbest.model'),
                                             map_location=lambda storage, loc: storage))
        self.nli_model = NLI()
        self.utt_vocab = utt_vocab

    def reward(self, hyp, ref, context, step_size):
        """
        hyp:     List(batch_size, seq_len)
        ref:     List(batch_size, seq_len)
        context: List(batch_size, seq_len)
        """
        hyp = self.repadding(hyp)
        context_decoded = [self.text_postprocess(' '.join([self.utt_vocab.id2word[token] for token in sentence])) for sentence in context]
        hyp_decoded = [self.text_postprocess(' '.join([self.utt_vocab.id2word[token] for token in sentence])) for sentence in hyp]

        ssn_pred = self.ssn_model.predict(XTarget=[torch.tensor(context).cuda(), torch.tensor(hyp).cuda()], DATarget=None, step_size=step_size)
        # ssn_pred: "probability of misordered", Tensor(batch_size), scalability=[0,1]
        nli_pred = self.nli_model.predict(x1=context_decoded, x2=hyp_decoded)
        # nli_pred: "probabilities of [entailment, neutral, contradiction]", List(batch_size, 3), scalability=[0,1]
        nli_pred = torch.tensor(nli_pred[:, 2]).cuda()
        reward = (1 - ssn_pred) + (1 - nli_pred)
        return reward

    def text_postprocess(self, text):
        text = text.split('<EOS>')[0]
        text = re.sub(r'<BOS>', '', text)
        return text

    def repadding(self, T):
        for i in range(len(T)):
            try:
                EOS_idx = T[i].index(self.utt_vocab.word2id['<EOS>'])
                T[i] = T[i][:EOS_idx+1] + [self.utt_vocab.word2id['<PAD>']] * (len(T[i])-EOS_idx-1)
            except:
                pass
        return T

def train(experiment, fine_tuning=False):
    print('loading setting "{}"...'.format(experiment))
    config = initialize_env(experiment)
    X_train, Y_train, XU_train, YU_train, turn_train = create_traindata(config=config, prefix='train')
    X_valid, Y_valid, XU_valid, YU_valid, turn_valid= create_traindata(config=config, prefix='valid')
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

    # Tokenize
    X_train, Y_train = da_vocab.tokenize(X_train), da_vocab.tokenize(Y_train)
    X_valid, Y_valid = da_vocab.tokenize(X_valid), da_vocab.tokenize(Y_valid)
    XU_train, YU_train = utt_vocab.tokenize(XU_train), utt_vocab.tokenize(YU_train)
    XU_valid, YU_valid = utt_vocab.tokenize(XU_valid), utt_vocab.tokenize(YU_valid)
    assert len(X_train) == len(Y_train), 'Unexpect content in train data'
    assert len(X_valid) == len(Y_valid), 'Unexpect content in valid data'

    lr = config['lr']
    batch_size = config['BATCH_SIZE']
    plot_losses = []
    print_total_loss = 0
    plot_total_loss = []
    reward_fn = Reward(utt_vocab=utt_vocab, da_vocab=da_vocab, config=config) if config['RL'] else None
    model = RL(utt_vocab=utt_vocab,
               da_vocab=da_vocab,
               fine_tuning=fine_tuning,
               reward_fn=reward_fn,
               criterion=nn.CrossEntropyLoss(ignore_index=utt_vocab.word2id['<PAD>'], reduce=False),
               config=config).cuda()
    if fine_tuning:
        model.load_state_dict(torch.load(os.path.join(config['log_root'], config['pretrain_expr'], 'statevalidbest.model'.format())))
    model_opt = optim.Adam(list(filter(lambda x: x.requires_grad, model.parameters())), lr=lr)
    print('Success construct model...')

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
            model_opt.zero_grad()
            # create batch data
            print('\rConversation {}/{} training...'.format(k + step_size, len(X_train)), end='')
            XU_seq = [XU_train[seq_idx] for seq_idx in batch_idx]
            YU_seq = [YU_train[seq_idx] for seq_idx in batch_idx]
            max_conv_len = max(len(s) for s in XU_seq)
            XU_tensor = []
            for i in range(0, max_conv_len):
                max_xseq_len = max(len(XU[i]) + 1 for XU in XU_seq)
                max_yseq_len = max(len(YU[i]) + 1 for YU in YU_seq)

                # utterance padding
                for ci in range(len(XU_seq)):
                    XU_seq[ci][i] = XU_seq[ci][i] + [utt_vocab.word2id['<PAD>']] * (max_xseq_len - len(XU_seq[ci][i]))
                    YU_seq[ci][i] = YU_seq[ci][i] + [utt_vocab.word2id['<PAD>']] * (max_yseq_len - len(YU_seq[ci][i]))
                XU_tensor.append(torch.tensor([XU[i] for XU in XU_seq]).cuda())
            YU_tensor= torch.tensor([YU[-1] for YU in YU_seq]).cuda()

            loss, _, _ = model.forward(X_utt=XU_tensor, Y_utt=YU_tensor, step_size=step_size)
            print_total_loss += loss
            plot_total_loss.append(loss)
            model_opt.step()
            k += step_size

        print()
        valid_loss, valid_reward = validation(XU_valid=XU_valid, YU_valid=YU_valid, model=model, utt_vocab=utt_vocab, config=config)

        def save_model(filename):
            torch.save(model.state_dict(), os.path.join(config['log_dir'], 'state{}.model'.format(filename)))

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


def validation(XU_valid, YU_valid, model, utt_vocab, config):
    model.eval()
    total_loss = 0
    k = 0
    batch_size = config['BATCH_SIZE']
    indexes = [i for i in range(len(XU_valid))]
    while k < len(indexes):
        step_size = min(batch_size, len(indexes) - k)
        batch_idx = indexes[k: k + step_size]
        XU_seq = [XU_valid[seq_idx] for seq_idx in batch_idx]
        YU_seq = [YU_valid[seq_idx] for seq_idx in batch_idx]
        max_conv_len = max(len(s) for s in XU_seq)
        XU_tensor = []
        YU_tensor = []
        for i in range(0, max_conv_len):
            max_xseq_len = max(len(XU[i]) + 1 for XU in XU_seq)
            max_yseq_len = max(len(YU[i]) + 1 for YU in YU_seq)
            for ci in range(len(XU_seq)):
                XU_seq[ci][i] = XU_seq[ci][i] + [utt_vocab.word2id['<PAD>']] * (max_xseq_len - len(XU_seq[ci][i]))
                YU_seq[ci][i] = YU_seq[ci][i] + [utt_vocab.word2id['<PAD>']] * (max_yseq_len - len(YU_seq[ci][i]))
            XU_tensor.append(torch.tensor([x[i] for x in XU_seq]).cuda())
            YU_tensor.append(torch.tensor([y[i] for y in YU_seq]).cuda())
        loss, reward, pred_seq = model.forward(X_utt=XU_tensor, Y_utt=YU_tensor, step_size=step_size)
        total_loss += loss
        if k == 0:
            sample_idx = random.sample([i for i in range(len(XU_seq))], 3)
            for idx in sample_idx:
                context = ' '.join([utt_vocab.id2word[wid] for wid in XU_seq[idx][-1]])
                context = context.split('<EOS>')[0]
                context = re.sub(r'<BOS>', '', context)
                hyp = ' '.join([utt_vocab.id2word[wid] for wid in pred_seq[idx]])
                hyp = hyp.split('<EOS>')
                hyp = re.sub(r'<BOS>', '', hyp)
                print('context:\t{}'.format(context))
                print('hyp:\t{}'.format(hyp))
        k += step_size
    return total_loss, reward

if __name__ == '__main__':
    args = parse()
    fine_tuning = False if 'pretrain' in args.expr else True
    train(args.expr, fine_tuning=fine_tuning)
