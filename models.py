import torch
import torch.nn as nn
import torch.nn.functional as F
from nn_blocks import *
from queue import PriorityQueue
import operator
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction


class RL(nn.Module):
    def __init__(self, utt_vocab, utt_encoder, utt_context, utt_decoder, config):
        super(RL, self).__init__()
        self.utt_vocab = utt_vocab
        self.utt_encoder = utt_encoder
        self.utt_context = utt_context
        self.utt_decoder = utt_decoder
        self.config = config

    def forward(self, X_utt, Y_utt, utt_context_hidden, step_size, criterion, last):
        """
        :param X_utt: context utterance tensor (batch_size, seq_len, 1)
        :param Y_utt: reference utterance tensor (batch_size, seq_len, 1)
        :param utt_context_hidden: hidden of context encoder (batch_size, 1, context_hidden_size)
        :param step_size: batch size (Scalar)
        :param criterion: loss function
        :param last: flg of conversation's last
        :return: loss, updated utt_context_hidden
        """
        CE_loss = 0
        utt_dec_hidden = self._encoding(X_utt=X_utt, utt_context_hidden=utt_context_hidden, step_size=step_size)

        # Response Decode
        # Greedy Decoding
        # pred_seq = []
        # for j in range(len(Y_utt[0]) - 1):
        #     prev_words = Y_utt[:, j].unsqueeze(1)
        #     logits, utt_decoder_hidden, utt_decoder_output = self.utt_decoder(prev_words, utt_dec_hidden)
        #     _, topi = logits.topk(1)
        #     pred_seq.append(topi.item())
        #     CE_loss += criterion(logits.view(-1, len(self.utt_vocab.word2id)), Y_utt[:, j + 1])

        # Decoding
        # TODO: not teacher forcing but inference
        pred_seq = []
        base_seq = []
        for j in range(len(Y_utt[0]) - 1):
            prev_words = Y_utt[:, j].unsqueeze(1)
            logits, decoder_hidden, _ = self.utt_decoder(prev_words, utt_dec_hidden)
            filtered_logits = self.top_k_top_p_filtering(logits=logits, top_k=self.config['top_k'],
                                                         top_p=self.config['top_p'])
            probs = F.softmax(filtered_logits, dim=-1)
            _, base_topi = logits.topk(1)
            next_token = torch.multinomial(probs, 1).squeeze(-1)
            pred_seq.append(next_token)
            base_seq.append(base_topi)
            CE_loss += criterion(probs.view(-1, len(self.utt_vocab.word2id)), Y_utt[:, j+1])
        pred_seq = torch.stack(pred_seq)
        base_seq = torch.stack(base_seq)
        
        # get reward
        pred_seq = [s for s in pred_seq.transpose(0,1).data.tolist()]
        base_seq = [[w[0] for w in s] for s in base_seq.transpose(0,1).data.tolist()]
        ref_seq = [s for s in Y_utt.data.tolist()]
        context = [s for s in X_utt.data.tolist()]
        reward = torch.tensor(self.reward(pred_seq, ref_seq, context)).cuda()
        b = torch.tensor(self.reward(base_seq, ref_seq, context)).cuda()

        # Optimized with REINFORCE
        RL_loss = CE_loss * (reward - b)
        # pg_loss = pg_loss.mean()
        pg_loss = CE_loss * self.config['lambda'] + RL_loss * (1 - self.config['lambda'])
        pg_loss = pg_loss.mean()
        if self.training:
            if last:
                pg_loss.backward()
                return pg_loss.item(), utt_context_hidden, reward
            else:
                return pg_loss, utt_context_hidden, reward
        else:
            return pg_loss.item(), utt_context_hidden, reward

    def reward(self, hypothesis, reference, context):
        # TODO: 報酬を考える
        r_bleu = self.calc_bleu(reference, hypothesis)
        return r_bleu

    def calc_bleu(self, refs, hyps):
        # refs = [' '.join(list(map(str, ref))) for ref in refs]
        # hyps = [' '.join(list(map(str, hyp))) for hyp in hyps]
        # bleu = get_moses_multi_bleu(hyps, refs, lowercase=True)
        # if bleu is None: bleu = 0.0
        refs = [[list(map(str, ref))] for ref in refs]
        hyps = [list(map(str, hyp)) for hyp in hyps]
        try:
            bleu = corpus_bleu(refs, hyps, smoothing_function=SmoothingFunction().method2)
        except:
            bleu = 1e-10
        return bleu

    def predict(self, X_utt, utt_context_hidden):
        with torch.no_grad():
            utt_decoder_hidden = self._encoding(X_utt=X_utt, utt_context_hidden=utt_context_hidden, step_size=1)
            prev_words = torch.tensor([[self.utt_vocab.word2id['<BOS>']]]).cuda()
            if self.config['beam_size']:
                pred_seq, utt_decoder_hidden = self._beam_decode(decoder=self.utt_decoder,
                                                                 decoder_hiddens=utt_decoder_hidden,
                                                                 config=self.config)
                pred_seq = pred_seq[0]
            else:
                pred_seq, utt_decoder_hidden = self._greedy_decode(prev_words, self.utt_decoder, utt_decoder_hidden)
        return pred_seq, utt_context_hidden

    def _encoding(self, X_utt, utt_context_hidden, step_size):
        # Encode Utterance
        utt_encoder_hidden = self.utt_encoder.initHidden(step_size)
        utt_encoder_output, utt_encoder_hidden = self.utt_encoder(X_utt, utt_encoder_hidden)  # (batch_size, 1, UTT_HIDDEN)

        # Update Context Encoder
        utt_context_output, utt_context_hidden = self.utt_context(utt_encoder_output, utt_context_hidden) # (batch_size, 1, UTT_CONTEXT)

        return utt_context_hidden

    def _greedy_decode(self, prev_words, decoder, decoder_hidden):
        EOS_token = self.utt_vocab.word2id['<EOS>']
        pred_seq = []
        for _ in range(self.config['max_len']):
            preds, decoder_hidden, _ = decoder(prev_words, decoder_hidden)
            _, topi = preds.topk(1)
            pred_seq.append(topi.item())
            prev_words = torch.tensor([[topi]]).cuda()
            if topi == EOS_token:
                break
        return pred_seq, decoder_hidden

    def _sample_decode(self, prev_words, decoder, decoder_hidden):
        EOS_token = self.utt_vocab.word2id['<EOS>']
        pred_seq = []
        for _ in range(self.config['max_len']):
            logits, decoder_hidden, _ = decoder(prev_words, decoder_hidden)
            filtered_logits = self.top_k_top_p_filtering(logits=logits, top_k=self.config['top_k'], top_p=self.config['top_p'])
            probs = F.softmax(filtered_logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            pred_seq.append(next_token)
            prev_words = torch.tensor(next_token).cuda()
            if next_token == EOS_token:
                break
        return pred_seq, decoder_hidden

    def top_k_top_p_filtering(self, logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
        """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
            Args:
                logits: logits distribution shape (vocabulary size)
                top_k >0: keep only top k tokens with highest probability (top-k filtering).
                top_p >0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                    Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        """
        top_k = min(top_k, logits.size(-1))  # Safety check
        if top_k > 0:
            # Remove all tokens with a probability less than the last token of the top-k
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = filter_value

        if top_p > 0.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs >= top_p
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = torch.zeros_like(logits, dtype=torch.uint8).scatter_(
                dim=-1, index=sorted_indices, src=sorted_indices_to_remove)
            logits[indices_to_remove] = filter_value
        return logits

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
            decoder_input = torch.tensor([[BOS_token]]).cuda()

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
    def __init__(self, utt_vocab,
                 utt_encoder, utt_context, utt_decoder, config):
        super(HRED, self).__init__()
        self.utt_vocab = utt_vocab
        self.utt_encoder = utt_encoder
        self.utt_context = utt_context
        self.utt_decoder = utt_decoder
        self.config = config

    def forward(self,X_utt, Y_utt, step_size,
                utt_context_hidden,
                criterion, last, baseline_outputs=None):
        loss = 0

        # Encoding
        utt_dec_hidden = self._encoding(X_utt=X_utt, utt_context_hidden=utt_context_hidden, step_size=step_size)

        # Response Decode
        for j in range(len(Y_utt[0]) - 1):
            prev_words = Y_utt[:, j].unsqueeze(1)
            preds, utt_decoder_hidden, utt_decoder_output = self.utt_decoder(prev_words, utt_dec_hidden)
            _, topi = preds.topk(1)
            loss += criterion(preds.view(-1, len(self.utt_vocab.word2id)), Y_utt[:, j + 1])
        # Calc. loss
        loss = loss.mean()
        if self.training:
            if last:
                loss.backward()
                return loss.item(), utt_context_hidden, 0
            else:
                return loss, utt_context_hidden, 0
        else:
            return loss.item(), utt_context_hidden, 0

    def predict(self, X_utt, utt_context_hidden, step_size=1):
        with torch.no_grad():
            utt_dec_hidden = self._encoding(X_utt=X_utt, utt_context_hidden=utt_context_hidden, step_size=step_size)

            utt_decoder_hidden = utt_dec_hidden
            prev_words = torch.tensor([[self.utt_vocab.word2id['<BOS>']] for _ in range(step_size)]).cuda()

            # if self.config['beam_size']:
            #     pred_seq, utt_decoder_hidden = self._beam_decode(decoder=self.utt_decoder, decoder_hiddens=utt_decoder_hidden, tag=tag, config=self.config)
            #     pred_seq = pred_seq[0]
            # else:
            pred_seq, utt_decoder_hidden = self._greedy_decode(prev_words, self.utt_decoder, utt_decoder_hidden, config=self.config)

        return pred_seq, utt_context_hidden

    def _encoding(self, X_utt, utt_context_hidden, step_size):
        # Encode Utterance
        utt_encoder_hidden = self.utt_encoder.initHidden(step_size)
        utt_encoder_output, utt_encoder_hidden = self.utt_encoder(X_utt, utt_encoder_hidden)  # (batch_size, 1, UTT_HIDDEN)

        # Update Context Encoder
        utt_context_output, utt_context_hidden = self.utt_context(utt_encoder_output, utt_context_hidden) # (batch_size, 1, UTT_HIDDEN)

        return utt_context_hidden

    def _greedy_decode(self, prev_words, decoder, decoder_hidden, config):
        PAD_token = self.utt_vocab.word2id['<PAD>']
        pred_seq = []
        for _ in range(config['max_len']):
            preds, decoder_hidden, _ = decoder(prev_words, decoder_hidden)
            _, topi = preds.topk(1)
            pred_seq.append(topi)
            prev_words = topi.clone()
            if all(ele == PAD_token for ele in topi):
                break
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

            decoder_input = torch.tensor([[BOS_token]]).cuda()

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

    def calc_bleu(self, refs, hyps):
        return corpus_bleu([[ref] for ref in refs], hyps)


class seq2seq(nn.Module):
    def __init__(self):
        super(seq2seq, self).__init__()

    def forward(self, X, Y, encoder, decoder, context, step_size, criterion, config):
        loss = 0

        encoder_hidden = encoder.initHidden(step_size)
        encoder_output, encoder_hidden = encoder(X, encoder_hidden)

        context_hidden = context.initHidden(step_size)
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
            encoder_hidden = encoder.initHidden(1)
            encoder_output, _ = encoder(X, encoder_hidden)

            context_hidden = context.initHidden(1)
            context_output, context_hidden = context(encoder_output, context_hidden)
            
            decoder_hidden = context_hidden
            prev_words = torch.tensor([[BOS_token]]).cuda()
            pred_seq = []
            for _ in range(config['max_len']):
                preds, decoder_hidden = decoder(prev_words, decoder_hidden)
                _, topi = preds.topk(1)
                pred_seq.append(topi.item())
                prev_words = torch.tensor([[topi]]).cuda()
                if topi == EOS_token:
                    break
        return pred_seq



