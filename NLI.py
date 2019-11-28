import torch
import torch.nn as nn
from torch import optim
from train import initialize_env, parse
import time, random
from transformers import *
from utils import *
from sklearn.metrics import accuracy_score

class Classifier(nn.Module):
    def __init__(self, encoder_hidden,middle_layer_size):
        super(Classifier, self).__init__()
        self.encoder_hidden = encoder_hidden
        self.middle_layer_size = middle_layer_size

        self.hm = nn.Linear(self.encoder_hidden, self.middle_layer_size)
        self.my = nn.Linear(self.middle_layer_size, 3)

    def forward(self, X):
        return self.my(self.hm(X))

class NLI(nn.Module):
    def __init__(self, classifier, criterion, config):
        super(NLI, self).__init__()
        self.classifier = classifier
        self.config = config
        self.tokenizer = BertTokenizer.from_pretrained(config['BERT_MODEL'])
        self.encoder = BertForSequenceClassification.from_pretrained(config['BERT_MODEL'],
                                                                           output_hidden_states=True,
                                                                           output_attentions=True)
        self.criterion = criterion
    def forward(self, X, Y):
        x_hidden, x_attentions = self.encoder(X)[-2:]
        x_hidden = x_hidden[-1]
        output = x_hidden[:, 0, :] # use [CLS] label representation
        pred = self.classifier(output)
        loss = self.criterion(pred, Y)
        loss.backward()
        return loss.item(), pred

    def predict(self, X):
        x_hidden, x_attentions = self.encoder(X)[-2:]
        x_hidden = [-1]
        output = x_hidden[:, 0, :]
        pred = self.classifier(output)
        return pred

def train(experiment):
    start = time.time()
    config = initialize_env(experiment)
    X, Y = NLILoader(config=config, prefix='train')
    X_valid, Y_valid = NLILoader(config=config, prefix='dev')
    criterion = nn.CrossEntropyLoss()
    lr = config['lr']
    batch_size = config['BATCH_SIZE']
    classifier = Classifier(encoder_hidden=768, middle_layer_size=config['NLI_MIDDLE_LAYER']).cuda()
    model = NLI(classifier=classifier, criterion=criterion, config=config).cuda()
    model_opt = optim.Adam(model.parameters(), lr=lr)
    _valid_loss = None
    early_stop = 0
    print_total_loss = 0
    for e in range(config['EPOCH']):
        tmp_time = time.time()
        print('Epoch {} start.'.format(e+1))
        indexes = [i for i in range(len(X))]
        random.shuffle(indexes)
        k = 0
        train_acc = []
        model.train()
        while k < len(indexes):
            step_size = min(batch_size, len(indexes)-k)
            print('\rTRAINING|\t{} / {} .'.format(k+step_size, len(indexes)), end='')
            batch_idx = indexes[k: k+step_size]
            model_opt.zero_grad()
            x = [X[i] for i in batch_idx]
            y = [Y[i] for i in batch_idx]
            x = string2tensor(model.tokenizer, list(x))
            y = torch.tensor(y).cuda()
            loss, pred = model(X=x, Y=y)
            preds = torch.argmax(pred, dim=1).data.tolist()
            train_acc.append(accuracy_score(y_true=y.data.tolist(), y_pred=preds))
            model_opt.step()
            k += step_size
            print_total_loss += loss
        print()
        valid_loss = validation(model, X_valid, Y_valid, config)
        def save(fname):
            torch.save(model.state_dict(), os.path.join(config['log_dir'], 'state{}.model'.format(fname)))
        if _valid_loss is None:
            save('validbest')
            _valid_loss = valid_loss
        else:
            if _valid_loss > valid_loss:
                save('validbest')
                early_stop = 0
            else:
                early_stop += 1
                if early_stop >= config['EARLY_STOP']: break

        if (e + 1) % config['LOGGING_FREQ'] == 0:
            print_loss_avg = print_total_loss / config['LOGGING_FREQ']
            print_total_loss = 0
            print('train acc. | ', np.mean(train_acc))
            print('epoch %d\tloss %.4f\tvalid loss %.4f\t | exec time %.4f' % (e + 1, print_loss_avg, valid_loss, time.time() - tmp_time))

        if (e + 1) % config['SAVE_MODEL'] == 0:
            print('save model')
            save(e+1)
    print()
    print('Finish training | exec time: %.4f [sec]' % (time.time() - start))

    
def validation(model, X, Y, config):
    batch_size = config['BATCH_SIZE']
    indexes = [i for i in range(len(X))]
    random.shuffle(indexes)
    k = 0
    model.eval()
    total_loss = 0
    val_acc = []
    while k < len(indexes):
        step_size = min(batch_size, len(indexes) - k)
        batch_idx = indexes[k: k + step_size]
        x = [X[i] for i in batch_idx]
        y = [Y[i] for i in batch_idx]
        x = string2tensor(model.tokenizer, x)
        y = torch.tensor(y).cuda()
        loss, pred = model(X=x, Y=y)
        preds = torch.argmax(pred, dim=1).data.tolist()
        val_acc.append(accuracy_score(y_true=y.data.tolist(), y_pred=preds))
        k += step_size
        total_loss += loss
    print('Avg. acc.: ', np.mean(val_acc))
    return loss


def evaluate(experiment):
    config = initialize_env(experiment)
    X, Y = NLILoader(config=config, prefix='test')
    criterion = nn.CrossEntropyLoss()
    batch_size = config['BATCH_SIZE']
    classifier = Classifier(encoder_hidden=768, middle_layer_size=config['NLI_MIDDLE_LAYER']).cuda()
    model = NLI(classifier=classifier, criterion=criterion, config=config).cuda()
    model.load_state_dict(torch.load(os.path.join(config['log_dir'], 'statevalidbest.model')))
    indexes = [i for i in range(X)]
    k = 0
    acc = []
    model.eval()
    while k < len(indexes):
        step_size = min(batch_size, len(indexes)-k)
        print('\rEVALUATION|\t{} / {} .'.format(k+step_size, len(indexes)), end='')
        batch_idx = indexes[k: k+step_size]
        x = [X[i] for i in batch_idx]
        y = [Y[i] for i in batch_idx]
        x = string2tensor(model.tokenizer, list(x))
        y = torch.tensor(y).cuda()
        loss, pred = model(X=x, Y=y)
        preds = torch.argmax(pred, dim=1).data.tolist()
        acc.append(accuracy_score(y_true=y.data.tolist(), y_pred=preds))
        k += step_size
    print()
    print('Avg. acc.: ', np.mean(acc))


def string2tensor(tokenizer, X):
    max_seq_len = max(len(batch) for batch in X)
    for bidx in range(len(X)):
        x_seq = tokenizer.encode(X[bidx])
        pad_len = max_seq_len - len(x_seq)
        X[bidx] = x_seq + [0 for _ in range(pad_len)]
        assert len(X[bidx]) == max_seq_len, '{}, {}'.format(len(X[bidx]), max_seq_len)
    return torch.tensor(X).cuda()

if __name__ == '__main__':
    args = parse()
    train(args.expr)
