import torch
import torch.nn as nn
from nn_blocks import *
from queue import PriorityQueue
import operator
from nltk.translate.bleu_score import corpus_bleu


class RL(nn.Module):
    def __init__(self, utt_vocab, utt_encoder, utt_context, utt_decoder, device, config):
        super(RL, self).__init__()
        self.device = device
        self.utt_vocab = utt_vocab
        self.utt_encoder = utt_encoder
        self.utt_context = utt_context
        self.utt_decoder = utt_decoder
        self.config = config

    def forward(self, X_utt, Y_utt, utt_context_hidden, step_size, criterion, last):
        CE_loss = 0
        utt_dec_hidden = self._encoding(X_utt=X_utt, utt_context_hidden=utt_context_hidden, step_size=step_size)

        # Response Decode
        pred_seq = []
        for j in range(len(Y_utt[0]) - 1):
            prev_words = Y_utt[:, j].unsqueeze(1)
            preds, utt_decoder_hidden, utt_decoder_output = self.utt_decoder(prev_words, utt_dec_hidden)
            _, topi = preds.topk(1)
            pred_seq.append(topi.item())
            CE_loss += criterion(preds.view(-1, len(self.utt_vocab.word2id)), Y_utt[:, j + 1])

        # Calc. reward
        reward = self.reward()

        pg_loss = CE_loss * reward
        pg_loss = pg_loss.mean()

        if self.training:
            if last:
                pg_loss.backward()
                return pg_loss.item(), utt_context_hidden
            else:
                return pg_loss, utt_context_hidden
        else:
            return pg_loss.item(), utt_context_hidden

    def reward(self, hypothesis, context):
        return 1.0

    def predict(self, X_utt, utt_context_hidden):
        with torch.no_grad():
            da_dec_hidden, utt_dec_hidden = self._encoding(X_utt=X_utt, utt_context_hidden=utt_context_hidden, step_size=1)

            utt_decoder_hidden = utt_dec_hidden
            prev_words = torch.tensor([[self.utt_vocab.word2id['<BOS>']]]).to(self.device)

            if self.config['beam_size']:
                pred_seq, utt_decoder_hidden = self._beam_decode(decoder=self.utt_decoder,
                                                                 decoder_hiddens=utt_decoder_hidden,
                                                                 config=self.config)
                pred_seq = pred_seq[0]
            else:
                pred_seq, utt_decoder_hidden = self._greedy_decode(prev_words, self.utt_decoder, utt_decoder_hidden,
                                                                   config=self.config)

        return pred_seq, utt_context_hidden

    def _encoding(self, X_utt, utt_context_hidden, step_size):
        # Encode Utterance
        utt_encoder_hidden = self.utt_encoder.initHidden(step_size, self.device)
        utt_encoder_output, utt_encoder_hidden = self.utt_encoder(X_utt, utt_encoder_hidden)  # (batch_size, 1, UTT_HIDDEN)

        # Update Context Encoder
        utt_context_output, utt_context_hidden = self.utt_context(utt_encoder_output, utt_context_hidden) # (batch_size, 1, UTT_CONTEXT)

        return utt_context_hidden

    def _greedy_decode(self, prev_words, decoder, decoder_hidden, config):
        EOS_token = self.utt_vocab.word2id['<EOS>']
        pred_seq = []
        for _ in range(config['max_len']):
            preds, decoder_hidden, _ = decoder(prev_words, decoder_hidden)
            _, topi = preds.topk(1)
            pred_seq.append(topi.item())
            prev_words = torch.tensor([[topi]]).to(self.device)
            if topi == EOS_token:
                break
        if not pred_seq[-1] == EOS_token: pred_seq.append(EOS_token)
        return pred_seq, decoder_hidden


    def _beam_decode(self, decoder, decoder_hiddens, config, encoder_outputs=None):
        BOS_token = self.utt_vocab.word2id['<BOS>']
        EOS_token = self.utt_vocab.word2id['<EOS>']
        decoded_batch = []
        topk = 1
        # batch対応
        for idx in range(decoder_hiddens.size(1)):
            if isinstance(decoder_hiddens, tuple):
                decoder_hidden = (decoder_hiddens[0][:, idx, :].unsqueeze(0), decoder_hiddens[1][idx, :, :].unsqueeze(0))
            else:
                decoder_hidden = decoder_hiddens[:, idx, :].unsqueeze(0)

            # encoder_output = encoder_outputs[idx, :, :].unsqueeze(1)

            decoder_input = torch.tensor([[BOS_token]]).to(self.device)

            endnodes = []
            number_required = min((topk + 1), (topk - len(endnodes)))

            node = BeamNode(hidden=decoder_hidden, previousNode=None, wordId=decoder_input, logProb=0, length=1)
            nodes = PriorityQueue()

            nodes.put((-node.eval(), node))
            qsize = 1

            while 1:
                if qsize > 2000: break

                score, n = nodes.get()
                decoder_input = n.wordid
                decoder_hidden = n.hidden

                if n.wordid.item() == EOS_token and n.prevNode != None:
                    endnodes.append((score, n))

                    if len(endnodes) >= number_required:
                        break
                    else:
                        continue

                decoder_output, decoder_hidden, _ = decoder(decoder_input, decoder_hidden)
                log_prob, indexes = torch.topk(decoder_output, config['beam_size'])
                nextnodes = []

                for new_k in range(config['beam_size']):
                    decoded_t = indexes[0][new_k].view(1, -1)
                    log_p = log_prob[0][new_k].item()

                    node = BeamNode(hidden=decoder_hidden, previousNode=n, wordId=decoded_t, logProb=n.logp + log_p, length=n.length + 1)
                    score = -node.eval()
                    nextnodes.append((score, node))

                for i in range(len(nextnodes)):
                    score, nn = nextnodes[i]
                    nodes.put((score, nn))
                qsize += len(nextnodes) - 1

            # choose nbest paths, back trace them
            if len(endnodes) == 0:
                endnodes = [nodes.get() for _ in range(topk)]

            pred_seq = []
            for score, n in sorted(endnodes, key=operator.itemgetter(0)):
                seq = []
                seq.append(n.wordid)

                while n.prevNode != None:
                    n = n.prevNode
                    seq.append(n.wordid)

                seq = seq[::-1]
                pred_seq.append([word.item() for word in seq])
            if not pred_seq[-1] == EOS_token: pred_seq.append(EOS_token)
            decoded_batch.append(pred_seq)

        return decoded_batch[0], decoder_hidden




class HRED(nn.Module):
    def __init__(self, da_vocab, utt_vocab, device,
                 da_encoder, utt_encoder, da_context, utt_context,
                 da_decoder, utt_decoder, config):
        super(HRED, self).__init__()
        self.device = device
        self.da_vocab = da_vocab
        self.utt_vocab = utt_vocab
        self.da_encoder = da_encoder
        self.utt_encoder = utt_encoder
        self.da_context = da_context
        self.utt_context = utt_context
        self.da_decoder = da_decoder
        self.utt_decoder = utt_decoder
        self.config = config

    def forward(self, X_da, Y_da, X_utt, Y_utt, step_size,
                da_context_hidden, utt_context_hidden,
                criterion, da_criterion, last):
        loss = 0

        # Encoding
        da_dec_hidden, utt_dec_hidden = self._encoding(X_da=X_da, X_utt=X_utt,
                                                       da_context_hidden=da_context_hidden,
                                                       utt_context_hidden=utt_context_hidden,
                                                       step_size=step_size)

        # DA Decode
        utt_decoder_hidden = utt_dec_hidden
        if self.config['use_da']:
            if self.config['merge_dic']:
                prev_words = torch.tensor([[self.utt_vocab.word2id['<TAG>']] for _ in range(step_size)]).to(self.device)
                preds, utt_decoder_hidden, utt_decoder_output = self.utt_decoder(prev_words, utt_decoder_hidden)
                da_decoder_output = self.da_decoder(utt_decoder_output)
            else:
                da_decoder_output = self.da_decoder(da_dec_hidden)
            da_output = da_decoder_output.squeeze(1)
            Y_da = Y_da.squeeze(0)
            Y_da = Y_da.squeeze(1)
            da_loss = da_criterion(da_output, Y_da)
        
        tag = da_decoder_output if self.config['speaker'] else None
        # Response Decode
        for j in range(len(Y_utt[0]) - 1):
            prev_words = Y_utt[:, j].unsqueeze(1)
            preds, utt_decoder_hidden, utt_decoder_output = self.utt_decoder(prev_words, utt_decoder_hidden, tag)
            _, topi = preds.topk(1)
            loss += criterion(preds.view(-1, len(self.utt_vocab.word2id)), Y_utt[:, j + 1])

        # Calc. loss
        if self.config['use_da'] and not self.config['merge_dic']:
            loss = self._calc_loss(utt_loss=loss, da_loss=da_loss, true_y=Y_da, config=self.config)
        elif self.config['merge_dic']:
            loss = self._calc_loss(utt_loss=loss, da_loss=da_loss, true_y=Y_da, config=self.config)
        else:
            loss = loss.mean()
        if self.training:
            if last:
                loss.backward()
                return loss.item(), da_context_hidden, utt_context_hidden
            else:
                return loss, da_context_hidden, utt_context_hidden
        else:
            return loss.item(), da_context_hidden, utt_context_hidden


    def predict(self, X_da, X_utt, da_context_hidden, utt_context_hidden):
        with torch.no_grad():
            da_dec_hidden, utt_dec_hidden = self._encoding(X_da=X_da, X_utt=X_utt,
                                                           da_context_hidden=da_context_hidden,
                                                           utt_context_hidden=utt_context_hidden,
                                                           step_size=1)

            if self.config['use_da'] and not self.config['merge_dic']:
                decoder_output = self.da_decoder(da_dec_hidden)
            elif self.config['merge_dic']:
                prev_words = torch.tensor([[self.utt_vocab.word2id['<TAG>']]]).to(self.device)
                preds, utt_decoder_hidden, utt_decoder_output = self.utt_decoder(prev_words, utt_dec_hidden)
                decoder_output = self.da_decoder(utt_decoder_output)
            else:
                decoder_output = None

            utt_decoder_hidden = utt_dec_hidden
            prev_words = torch.tensor([[self.utt_vocab.word2id['<BOS>']]]).to(self.device)
            
            tag = decoder_output if self.config['speaker'] else None
            if self.config['beam_size']:
                pred_seq, utt_decoder_hidden = self._beam_decode(decoder=self.utt_decoder, decoder_hiddens=utt_decoder_hidden, tag=tag, config=self.config)
                pred_seq = pred_seq[0]
            else:
                pred_seq, utt_decoder_hidden = self._greedy_decode(prev_words, self.utt_decoder, utt_decoder_hidden, tag, config=self.config)

        return pred_seq, da_context_hidden, utt_context_hidden, decoder_output


    def _encoding(self, X_da, X_utt, da_context_hidden, utt_context_hidden, step_size):
        # Input Dialogue Act
        if self.config['use_da']:
            da_encoder_hidden = self.da_encoder(X_da)
            da_context_output, da_context_hidden = self.da_context(da_encoder_hidden, da_context_hidden)

        # Encode Utterance
        utt_encoder_hidden = self.utt_encoder.initHidden(step_size, self.device)
        utt_encoder_output, utt_encoder_hidden = self.utt_encoder(X_utt, utt_encoder_hidden)  # (batch_size, 1, UTT_HIDDEN)

        # Update Context Encoder
        utt_context_output, utt_context_hidden = self.utt_context(utt_encoder_output, utt_context_hidden) # (batch_size, 1, UTT_HIDDEN)

        # concat DA hidden and Utterance hidden
        if self.config['use_da']:
            da_dec_hidden = torch.cat((da_context_output, utt_context_output), dim=2)
            utt_dec_hidden = torch.cat((da_context_hidden, utt_context_hidden), dim=2) # (batch_size, 1, DEC_HIDDEN)
        else:
            da_dec_hidden = utt_context_output
            utt_dec_hidden = utt_context_hidden

        return da_dec_hidden, utt_dec_hidden


    def _calc_loss(self, utt_loss, da_loss, true_y, config):
        # for idx, y in enumerate(true_y):
        #     if y == self.da_vocab.word2id['<Uninterpretable>']:
        #         # da_loss[idx] = da_loss[idx] / 1e-10
        #         utt_loss[idx] = utt_loss[idx] / 1e-10
        # alpha = config['alpha']
        alpha = 0.5
        return (1 - alpha) * utt_loss.mean() + alpha * da_loss.mean()


    def _greedy_decode(self, prev_words, decoder, decoder_hidden, tag, config):
        EOS_token = self.utt_vocab.word2id['<EOS>']
        pred_seq = []
        for _ in range(config['max_len']):
            preds, decoder_hidden, _ = decoder(prev_words, decoder_hidden, tag)
            _, topi = preds.topk(1)
            pred_seq.append(topi.item())
            prev_words = torch.tensor([[topi]]).to(self.device)
            if topi == EOS_token:
                break
        if not pred_seq[-1] == EOS_token: pred_seq.append(EOS_token)
        return pred_seq, decoder_hidden


    def _beam_decode(self, decoder, decoder_hiddens, tag, config, encoder_outputs=None):
        BOS_token = self.utt_vocab.word2id['<BOS>']
        EOS_token = self.utt_vocab.word2id['<EOS>']
        decoded_batch = []
        topk = 1
        # batch対応
        for idx in range(decoder_hiddens.size(1)):
            if isinstance(decoder_hiddens, tuple):
                decoder_hidden = (decoder_hiddens[0][:, idx, :].unsqueeze(0), decoder_hiddens[1][idx, :, :].unsqueeze(0))
            else:
                decoder_hidden = decoder_hiddens[:, idx, :].unsqueeze(0)

            # encoder_output = encoder_outputs[idx, :, :].unsqueeze(1)

            decoder_input = torch.tensor([[BOS_token]]).to(self.device)

            endnodes = []
            number_required = min((topk + 1), (topk - len(endnodes)))

            node = BeamNode(hidden=decoder_hidden, previousNode=None, wordId=decoder_input, logProb=0, length=1)
            nodes = PriorityQueue()

            nodes.put((-node.eval(), node))
            qsize = 1

            while 1:
                if qsize > 2000: break

                score, n = nodes.get()
                decoder_input = n.wordid
                decoder_hidden = n.hidden

                if n.wordid.item() == EOS_token and n.prevNode != None:
                    endnodes.append((score, n))

                    if len(endnodes) >= number_required:
                        break
                    else:
                        continue

                decoder_output, decoder_hidden, _ = decoder(decoder_input, decoder_hidden, tag)
                log_prob, indexes = torch.topk(decoder_output, config['beam_size'])
                nextnodes = []

                for new_k in range(config['beam_size']):
                    decoded_t = indexes[0][new_k].view(1, -1)
                    log_p = log_prob[0][new_k].item()

                    node = BeamNode(hidden=decoder_hidden, previousNode=n, wordId=decoded_t, logProb=n.logp + log_p, length=n.length + 1)
                    score = -node.eval()
                    nextnodes.append((score, node))

                for i in range(len(nextnodes)):
                    score, nn = nextnodes[i]
                    nodes.put((score, nn))
                qsize += len(nextnodes) - 1

            # choose nbest paths, back trace them
            if len(endnodes) == 0:
                endnodes = [nodes.get() for _ in range(topk)]

            pred_seq = []
            for score, n in sorted(endnodes, key=operator.itemgetter(0)):
                seq = []
                seq.append(n.wordid)

                while n.prevNode != None:
                    n = n.prevNode
                    seq.append(n.wordid)

                seq = seq[::-1]
                pred_seq.append([word.item() for word in seq])
            if not pred_seq[-1] == EOS_token: pred_seq.append(EOS_token)
            decoded_batch.append(pred_seq)

        return decoded_batch[0], decoder_hidden

    def calc_bleu(self, refs, hyps):
        return corpus_bleu([[ref] for ref in refs], hyps)


class seq2seq(nn.Module):
    def __init__(self, device):
        super(seq2seq, self).__init__()
        self.device = device

    def forward(self, X, Y, encoder, decoder, context, step_size, criterion, config):
        loss = 0

        encoder_hidden = encoder.initHidden(step_size, self.device)
        encoder_output, encoder_hidden = encoder(X, encoder_hidden)

        context_hidden = context.initHidden(step_size, self.device)
        context_output, context_hidden = context(encoder_output, context_hidden)

        decoder_hidden = context_hidden
        for j in range(len(Y[0]) - 1):
            prev_words = Y[:, j].unsqueeze(1)
            preds, decoder_hidden = decoder(prev_words, decoder_hidden)
            _, topi = preds.topk(1)
            loss += criterion(preds.view(-1, config['UTT_MAX_VOCAB']), Y[:, j + 1])

        if self.training:
            loss.backward()

        return loss.item()

    def predict(self, X, encoder, decoder, context, config, EOS_token, BOS_token):
        with torch.no_grad():
            encoder_hidden = encoder.initHidden(1, self.device)
            encoder_output, _ = encoder(X, encoder_hidden)

            context_hidden = context.initHidden(1, self.device)
            context_output, context_hidden = context(encoder_output, context_hidden)
            
            decoder_hidden = context_hidden
            prev_words = torch.tensor([[BOS_token]]).to(self.device)
            pred_seq = []
            for _ in range(config['max_len']):
                preds, decoder_hidden = decoder(prev_words, decoder_hidden)
                _, topi = preds.topk(1)
                pred_seq.append(topi.item())
                prev_words = torch.tensor([[topi]]).to(self.device)
                if topi == EOS_token:
                    break
        return pred_seq



