import time
from torch import optim
from models import *
from utils import *
from nn_blocks import *
import random
from NLI import NLI
from order_predict import OrderPredictor
from DApredict import DApredictModel


class Reward:
    def __init__(self, utt_vocab, da_vocab, config, mode='train'):
        self.utt_vocab = utt_vocab
        self.da_vocab = da_vocab
        self.config = config
        self.mode = mode
        self.ssn_model = OrderPredictor(utt_vocab=utt_vocab, da_vocab=da_vocab, config=config).cuda()
        self.ssn_model.load_state_dict(torch.load(os.path.join(config['log_root'], 'order_predict', 'orderpred_statevalidbest.model'),
                                             map_location=lambda storage, loc: storage))
        self.nli_model = NLI()
        self.da_estimator = DApredictModel(utt_vocab=utt_vocab, da_vocab=da_vocab, config=config).cuda()
        self.da_estimator.load_state_dict(torch.load(os.path.join(config['log_root'], 'DAestimate', 'da_pred_statevalidbest.model'), map_location=lambda storage, loc: storage))
        self.da_predictor = DApredictModel(utt_vocab=utt_vocab, da_vocab=da_vocab, config=config).cuda()
        self.da_predictor.load_state_dict(torch.load(os.path.join(config['log_root'], 'DApredict_da', 'da_pred_statevalidbest.model'), map_location=lambda storage, loc: storage))


    def reward(self, hyp, ref, context, da_context, turn, step_size):
        """
        hyp:     List(batch_size, seq_len)
        ref:     List(batch_size, seq_len)
        context: List(window_size, batch_size, seq_len)
        da_context: List(window_size, batch_size, 1)
        turn:    List(window_size, batch_size, 1)
        step_size:  Scalar
        """
        self.rewards = {}
        # preprocess
        hyp = self.repadding(hyp)
        context_decoded = [[text_postprocess(' '.join([self.utt_vocab.id2word[token] for token in sentence])) for sentence in conv] for conv in context]
        hyp_decoded = [text_postprocess(' '.join([self.utt_vocab.id2word[token] for token in sentence])) for sentence in hyp]
        X_da = [torch.tensor(xda).cuda() for xda in da_context]

        # DA reward
        da_predicted = np.argmax(self.da_predictor.predict(X_da=X_da,
                                                           X_utt=[torch.tensor(sentence).clone().cuda() for sentence in context] + [torch.tensor(hyp).clone().cuda()],
                                                           turn=[torch.tensor(t).clone().cuda() for t in turn] + [torch.tensor([[1] for _ in range(step_size)]).clone().cuda()],
                                                           step_size=step_size), axis=1)
        self.rewards['da_pred'] = [self.da_vocab.id2word[t] for t in da_predicted]
        if self.config['NRG']['da_rwd'] or self.mode == 'test':
            da_candidate = self.da_estimator.predict(X_da=X_da, X_utt=[torch.tensor(sentence).clone().cuda() for sentence in context], turn=[torch.tensor(t).clone().cuda() for t in turn], step_size=step_size)
            # da_candidate: "probabilities of next DA", Numpy(batch_size, len(da_vocab)), scalability=[0,1]
            da_estimate_topk = np.argsort(da_candidate, axis=1)[:, -2:][::-1]
            da_rwd = []
            for bidx in range(len(da_candidate)):
                if da_predicted[bidx] in da_estimate_topk[bidx]:
                    da_rwd.append(da_candidate[bidx][da_predicted[bidx]])
                else:
                    da_rwd.append(0.0)
            da_rwd = torch.tensor(da_rwd).cuda()
            da_rwd = (da_rwd - self.config['zmean_da']) / self.config['zstd_da']
            self.rewards['da_rwd'] = da_rwd.data.tolist()
            self.rewards['da_estimate'] = [[self.da_vocab.id2word[t] for t in batch] for batch in da_estimate_topk]
        else:
            self.rewards['da_rwd'] = [0] * step_size
            da_rwd = 0

        # ordered reward
        if self.config['NRG']['ssn_rwd'] or self.mode == 'test':
            ssn_pred = self.ssn_model.predict(XTarget=[torch.tensor(sentence).clone().cuda() for sentence in context] + [torch.tensor(hyp).clone().cuda()], DATarget=[torch.tensor(da).clone().cuda() for da in da_context + [da_predicted]], step_size=step_size)
            ssn_pred = ((1 - ssn_pred) - self.config['zmean_ssn']) / self.config['zstd_ssn']
            # ssn_pred = 1 - ssn_pred
            # ssn_pred: "probability of misordered", Tensor(batch_size), scalability=[0,1]
            self.rewards['ssn'] = ssn_pred.data.tolist()
        else:
            self.rewards['ssn'] = [0] * step_size
            ssn_pred = 0

        # contradiction reward
        if self.config['NRG']['nli_rwd'] or self.mode == 'test':
            nli_preds = []
            for sentence in context_decoded:
                nli_pred = self.nli_model.predict(x1=sentence, x2=hyp_decoded)
                nli_pred = nli_pred[:, 2]
                nli_preds.append(nli_pred)
            nli_pred = torch.tensor(nli_preds).cuda().max(dim=0)[0]
            nli_pred = ((1 - nli_pred) - self.config['zmean_nli']) / self.config['zstd_nli']
            # nli_pred = 1 - nli_pred
            # nli_pred: "probabilities of [entailment, neutral, contradiction]", List(batch_size, 3), scalability=[0,1]
            self.rewards['nli'] = nli_pred.data.tolist()
        else:
            self.rewards['nli'] = [0] * step_size
            nli_pred = 0

        if self.config['assortment'] == 'summation':
            reward = (ssn_pred * self.config['NRG']['weight_ssn']) \
                     + (nli_pred * self.config['NRG']['weight_nli']) \
                     + (da_rwd * self.config['NRG']['weight_da'])
        elif self.config['assortment'] == 'geomean':
            reward = ((ssn_pred + 1e-5) * (nli_pred + 1e-5) * (da_rwd + 1e-5)) ** (1/3)
        return reward

    def repadding(self, T):
        for i in range(len(T)):
            try:
                EOS_idx = T[i].index(self.utt_vocab.word2id['<EOS>'])
                T[i] = T[i][:EOS_idx+1] + [self.utt_vocab.word2id['<PAD>']] * (len(T[i])-EOS_idx-1)
            except:
                pass
        return T

def train(args, fine_tuning=False):
    config = initialize_env(args.expr)
    X_train, Y_train, XU_train, YU_train, turn_train = create_traindata(config=config, prefix='train')
    X_valid, Y_valid, XU_valid, YU_valid, turn_valid= create_traindata(config=config, prefix='valid')
    print('Finish create train data...')

    if os.path.exists(os.path.join(config['log_root'], 'utterance_vocab.dict')):
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
    print_total_loss = 0
    reward_fn = Reward(utt_vocab=utt_vocab, da_vocab=da_vocab, config=config) if config['RL'] else None
    model = RL(utt_vocab=utt_vocab,
               da_vocab=da_vocab,
               fine_tuning=fine_tuning,
               reward_fn=reward_fn,
               criterion=nn.CrossEntropyLoss(ignore_index=utt_vocab.word2id['<PAD>'], reduce=False),
               config=config).cuda()
    if fine_tuning:
        model.load_state_dict(torch.load(os.path.join(config['log_root'], config['pretrain_expr'], 'statevalidbest.model'.format()), map_location=lambda storage, loc: storage))
    # if args.checkpoint:
    #     print('load checkpoint at epoch {}'.format(args.checkpoint))
    #     log_f = open(os.path.join(config['log_dir'], 'train_log.txt'), 'a')
    #     model.load_state_dict(torch.load(os.path.join(config['log_dir'], 'state{}.model'.format(args.checkpoint)), map_location=lambda storage, loc: storage))
    # else:
    #     log_f = open(os.path.join(config['log_dir'], 'train_log.txt'), 'w')
    model_opt = optim.Adam(list(filter(lambda x: x.requires_grad, model.parameters())), lr=lr)
    print('Success construct model...')

    print('---start training---')
    start = time.time()
    _monitor_score = None
    _train_loss = None
    early_stop = 0
    indexes = [i for i in range(len(X_train))]
    for e in range(config['EPOCH']):
        if args.checkpoint:
            e += int(args.checkpoint)
        tmp_time = time.time()
        print('Epoch {} start'.format(e+1))
        random.shuffle(indexes)
        k = 0
        model.train()
        rewards = []
        nli_rwds = []
        ssn_rwds = []
        da_rwds = []
        while k < len(indexes):
            # initialize
            step_size = min(batch_size, len(indexes) - k)
            batch_idx = indexes[k : k + step_size]
            model_opt.zero_grad()
            # create batch data
            print('\rConversation {}/{} training...'.format(k + step_size, len(X_train)), end='')
            XU_seq = [XU_train[seq_idx] for seq_idx in batch_idx]
            YU_seq = [YU_train[seq_idx] for seq_idx in batch_idx]
            XD_seq = [X_train[seq_idx] for seq_idx in batch_idx]
            turn_seq = [turn_train[seq_idx] for seq_idx in batch_idx]
            max_conv_len = max(len(s) for s in XU_seq)
            XU_tensor = []
            XD_tensor = []
            turn_tensor = []
            for i in range(0, max_conv_len):
                max_xseq_len = max(len(XU[i]) + 1 for XU in XU_seq)
                max_yseq_len = max(len(YU[i]) + 1 for YU in YU_seq)
                # utterance padding
                for ci in range(len(XU_seq)):
                    XU_seq[ci][i] = XU_seq[ci][i] + [utt_vocab.word2id['<PAD>']] * (max_xseq_len - len(XU_seq[ci][i]))
                    YU_seq[ci][i] = YU_seq[ci][i] + [utt_vocab.word2id['<PAD>']] * (max_yseq_len - len(YU_seq[ci][i]))
                XU_tensor.append(torch.tensor([XU[i] for XU in XU_seq]).cuda())
                XD_tensor.append(torch.tensor([[x[i]] for x in XD_seq]).cuda())
                turn_tensor.append(torch.tensor([[t[i]] for t in turn_seq]).cuda())
            YU_tensor= torch.tensor([YU[-1] for YU in YU_seq]).cuda()
            loss, reward, _ = model.forward(X_utt=XU_tensor, Y_utt=YU_tensor, X_da=XD_tensor, turn=turn_tensor, step_size=step_size)
            if config['RL']:
                nli_rwds.append(np.mean(reward_fn.rewards['nli']))
                ssn_rwds.append(np.mean(reward_fn.rewards['ssn']))
                da_rwds.append(np.mean(reward_fn.rewards['da_rwd']))
                # da_preds.append(reward_fn.rewards['da_pred'])
                # da_estis.append(reward_fn.rewards['da_estimate'])
            else:
                nli_rwds.append(0)
                ssn_rwds.append(0)
                da_rwds.append(0)
            rewards.append(reward)
            print_total_loss += loss
            model_opt.step()
            k += step_size
        print()
        nli_rwd = np.mean(nli_rwds)
        ssn_rwd = np.mean(ssn_rwds)
        da_rwd = np.mean(da_rwds)
        print('nli: {}, ssn: {}, da: {}'.format(nli_rwd, ssn_rwd, da_rwd))
        valid_loss, valid_reward, valid_bleu = validation(XU_valid=XU_valid, YU_valid=YU_valid, XD_valid=X_valid, turn_valid=turn_valid, model=model, utt_vocab=utt_vocab, config=config)
        # log_f.write('{},{},{},{},{},{},{},{},{}\n'.format(e + 1, print_total_loss, np.mean(rewards), valid_loss, valid_bleu, valid_reward, nli_rwd, ssn_rwd, da_rwd))
        def save_model(filename):
            torch.save(model.state_dict(), os.path.join(config['log_dir'], 'state{}.model'.format(filename)))

        if config['monitor'] == 'valid_bleu':
            monitor_score = valid_bleu
        elif config['monitor'] == 'valid_reward':
            monitor_score = valid_reward
        else:
            monitor_score = valid_loss

        if _monitor_score is None:
            save_model('validbest')
            _monitor_score = monitor_score
        else:
            if _monitor_score < monitor_score:
                save_model('validbest')
                _monitor_score = monitor_score
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
            print('train reward: {}, valid reward: {}'.format(np.mean(rewards), valid_reward))
            print('steps %d\tloss %.4f\tvalid loss %.4f\tvalid reward %.4f\tvalid bleu %.4f | exec time %.4f' % (e + 1, print_loss_avg, valid_loss, valid_reward, valid_bleu, time.time() - tmp_time))

        if (e + 1) % config['SAVE_MODEL'] == 0:
            print('saving model')
            save_model(e+1)

    print()
    # log_f.close()
    print('Finish training | exec time: %.4f [sec]' % (time.time() - start))


def validation(XU_valid, YU_valid, XD_valid, turn_valid, model, utt_vocab, config):
    model.eval()
    total_loss = 0
    k = 0
    batch_size = config['BATCH_SIZE']
    indexes = [i for i in range(len(XU_valid))]
    rewards = []
    bleus = []
    while k < len(indexes):
        step_size = min(batch_size, len(indexes) - k)
        batch_idx = indexes[k: k + step_size]
        XU_seq = [XU_valid[seq_idx] for seq_idx in batch_idx]
        YU_seq = [YU_valid[seq_idx] for seq_idx in batch_idx]
        XD_seq = [XD_valid[seq_idx] for seq_idx in batch_idx]
        turn_seq = [turn_valid[seq_idx] for seq_idx in batch_idx]
        max_conv_len = max(len(s) for s in XU_seq)
        XU_tensor = []
        XD_tensor = []
        turn_tensor = []
        for i in range(0, max_conv_len):
            max_xseq_len = max(len(XU[i]) + 1 for XU in XU_seq)
            max_yseq_len = max(len(YU[i]) + 1 for YU in YU_seq)
            for ci in range(len(XU_seq)):
                XU_seq[ci][i] = XU_seq[ci][i] + [utt_vocab.word2id['<PAD>']] * (max_xseq_len - len(XU_seq[ci][i]))
                YU_seq[ci][i] = YU_seq[ci][i] + [utt_vocab.word2id['<PAD>']] * (max_yseq_len - len(YU_seq[ci][i]))
            XU_tensor.append(torch.tensor([x[i] for x in XU_seq]).cuda())
            XD_tensor.append(torch.tensor([[x[i]] for x in XD_seq]).cuda())
            turn_tensor.append(torch.tensor([[t[i]] for t in turn_seq]).cuda())
        YU_tensor= torch.tensor([y[-1] for y in YU_seq]).cuda()
        loss, reward, pred_seq = model.forward(X_utt=XU_tensor, Y_utt=YU_tensor, X_da=XD_tensor, turn=turn_tensor, step_size=step_size)
        total_loss += loss
        bleus.append(corpus_bleu([[y] for y in YU_tensor.data.tolist()], pred_seq, smoothing_function=SmoothingFunction().method2))
        rewards.append(reward)
        if k == 0:
            sample_idx = random.sample([i for i in range(len(XU_seq))], 3)
            for idx in sample_idx:
                context = ' '.join([utt_vocab.id2word[wid] for wid in XU_seq[idx][-1]])
                context = context.split('<EOS>')[0]
                context = re.sub(r'<BOS>', '', context)
                hyp = ' '.join([utt_vocab.id2word[wid] for wid in pred_seq[idx]])
                hyp = hyp.split('<EOS>')[0]
                hyp = re.sub(r'<BOS>', '', hyp)
                print('context:\t{}'.format(context))
                print('hyp:\t{}'.format(hyp))
        k += step_size
    return total_loss, np.mean(rewards), np.mean(bleus)

if __name__ == '__main__':
    args = parse()
    fine_tuning = False if 'pretrain' in args.expr else True
    train(args, fine_tuning=fine_tuning)
