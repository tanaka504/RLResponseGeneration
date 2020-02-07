from nn_blocks import *
from utils import *
import operator
from queue import PriorityQueue


class RL(nn.Module):
    def __init__(self, utt_vocab, da_vocab, fine_tuning, reward_fn, criterion, config):
        super(RL, self).__init__()
        self.utt_vocab = utt_vocab
        self.da_vocab = da_vocab
        self.utt_encoder = UtteranceEncoder(utt_input_size=len(utt_vocab.word2id), embed_size=config['NRG']['UTT_EMBED'],
                                            utterance_hidden=config['NRG']['UTT_HIDDEN'], padding_idx=utt_vocab.word2id['<PAD>'],
                                            fine_tuning=fine_tuning).cuda()
        self.utt_decoder = UtteranceDecoder(utterance_hidden_size=config['NRG']['DEC_HIDDEN'], utt_embed_size=config['NRG']['UTT_EMBED'],
                                            utt_vocab_size=len(utt_vocab.word2id)).cuda()
        self.utt_context = UtteranceContextEncoder(utterance_hidden_size=config['NRG']['UTT_CONTEXT']).cuda()
        self.config = config
        self.reward_fn = reward_fn
        self.criterion = criterion

    def forward(self, X_utt, Y_utt, X_da, turn, step_size):
        """
        :param X_utt: context utterance tensor Tensor(window_size, batch_size, seq_len, 1)
        :param Y_utt: reference utterance tensor Tensor(batch_size, seq_len, 1)
        :param step_size: batch size Scalar(1)
        :return: loss, reward, predicted sequence
        """
        CE_loss = 0
        base_loss = 0
        decoder_hidden = self._encoding(X_utt=X_utt, step_size=step_size)

        # Decoding
        pred_seq = []
        base_seq = []
        for j in range(len(Y_utt[0]) - 1):
            prev_words = Y_utt[:, j].unsqueeze(1)
            logits, decoder_hidden, _ = self.utt_decoder(prev_words, decoder_hidden)
            filtered_logits = self.top_k_top_p_filtering(_logits=logits, top_k=self.config['NRG']['top_k'],
                                                         top_p=self.config['NRG']['top_p'])
            probs = F.softmax(filtered_logits, dim=-1)
            _, base_topi = logits.topk(1)
            next_token = torch.multinomial(probs, 1).squeeze(-1)
            pred_seq.append(next_token)
            base_seq.append(base_topi.squeeze(-1))
            if self.config['RL']:
                CE_loss += self.criterion(probs, Y_utt[:, j+1])
            else:
                base_loss += self.criterion(logits, Y_utt[:, j+1])

        if self.config['RL']:
            pred_seq = torch.stack(pred_seq)
            base_seq = torch.stack(base_seq)
            # get reward
            pred_seq = [s for s in pred_seq.transpose(0,1).data.tolist()]
            base_seq = [s for s in base_seq.transpose(0,1).data.tolist()]
            ref_seq = [s for s in Y_utt.data.tolist()]
            context = [[s for s in X.data.tolist()] for X in X_utt]
            da_context = [xd.data.tolist() for xd in X_da]
            turn = [t.data.tolist() for t in turn]
            reward = self.reward_fn.reward(hyp=pred_seq, ref=ref_seq, context=context, da_context=da_context, turn=turn, step_size=step_size)
            b = self.reward_fn.reward(hyp=base_seq, ref=ref_seq, context=context, da_context=da_context, turn=turn, step_size=step_size)

            # Optimized with REINFORCE
            RL_loss = CE_loss * (reward - b)
            loss = CE_loss * self.config['NRG']['lambda'] + RL_loss * (1 - self.config['NRG']['lambda'])
            loss = loss.mean()
            reward = reward.mean().item()
        else:
            reward = 0
            loss = base_loss.mean()
            pred_seq = torch.stack(base_seq).transpose(0, 1).data.tolist()
        if self.training:
            loss.backward()
        return loss.item(), reward, pred_seq

    def predict(self, X_utt, step_size):
        with torch.no_grad():
            utt_decoder_hidden = self._encoding(X_utt=X_utt, step_size=step_size)
            # if self.config['beam_size']:
            #     pred_seq, utt_decoder_hidden = self._beam_decode(decoder=self.utt_decoder,
            #                                                      decoder_hiddens=utt_decoder_hidden,
            #                                                      config=self.config)
            # else:
            pred_seq, utt_decoder_hidden = self._greedy_decode(utt_decoder_hidden, step_size)
        return torch.stack(pred_seq).transpose(0, 1).data.tolist()

    def perplexity(self, X, Y, step_size):
        with torch.no_grad():
            loss = 0
            decoder_hidden = self._encoding(X_utt=X, step_size=step_size)
            for j in range(len(Y[0]) - 1):
                prev_words = Y[:, j].unsqueeze(1)
                logits, decoder_hidden, _ = self.utt_decoder(prev_words, decoder_hidden)
                loss += self.criterion(logits, Y[:, j+1])
        return loss.data.tolist()

    def _encoding(self, X_utt, step_size):
        # Encode Utterance
        utt_context_hidden = self.utt_context.initHidden(step_size)
        for X in X_utt:
            utt_encoder_hidden = self.utt_encoder.initHidden(step_size)
            utt_encoder_output, utt_encoder_hidden = self.utt_encoder(X, utt_encoder_hidden)  # (batch_size, 1, UTT_HIDDEN)
            utt_encoder_output = torch.cat((utt_encoder_output[:, -1, :self.utt_encoder.hidden_size], utt_encoder_output[:, 0, self.utt_encoder.hidden_size:]), dim=-1).unsqueeze(1)
            # Update Context Encoder
            utt_context_output, utt_context_hidden = self.utt_context(utt_encoder_output, utt_context_hidden) # (batch_size, 1, UTT_CONTEXT)

        return utt_context_hidden

    def _greedy_decode(self, decoder_hidden, step_size):
        PAD_token = self.utt_vocab.word2id['<PAD>']
        prev_words = torch.tensor([[self.utt_vocab.word2id['<BOS>']] for _ in range(step_size)]).cuda()
        pred_seq = []
        for _ in range(self.config['max_len']):
            preds, decoder_hidden, _ = self.utt_decoder(prev_words, decoder_hidden)
            _, topi = preds.topk(1)
            pred_seq.append(topi.squeeze(-1))
            prev_words = topi.clone()
            if all(ele == PAD_token for ele in topi):
                break
        return pred_seq, decoder_hidden

    def _sample_decode(self, prev_words, decoder, decoder_hidden):
        PAD_token = self.utt_vocab.word2id['<PAD>']
        pred_seq = []
        for _ in range(self.config['max_len']):
            logits, decoder_hidden, _ = decoder(prev_words, decoder_hidden)
            filtered_logits = self.top_k_top_p_filtering(_logits=logits, top_k=self.config['top_k'], top_p=self.config['top_p'])
            probs = F.softmax(filtered_logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            pred_seq.append(next_token)
            prev_words = torch.tensor(next_token).cuda()
            if all(ele == PAD_token for ele in next_token):
                break
        return pred_seq, decoder_hidden

    def top_k_top_p_filtering(self, _logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
        """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
            Args:
                logits: logits distribution shape (vocabulary size)
                top_k >0: keep only top k tokens with highest probability (top-k filtering).
                top_p >0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                    Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        """
        logits = _logits.clone()
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
        return logits,

    def _beam_decode(self, decoder, decoder_hiddens, config):
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
        return decoded_batch, decoder_hidden


class seq2seq(nn.Module):
    def __init__(self, utt_vocab, da_vocab, reward_fn, config):
        super(seq2seq, self).__init__()
        self.criterion = nn.CrossEntropyLoss(ignore_index=utt_vocab.word2id['<PAD>'], reduce=False)
        self.utt_vocab = utt_vocab
        self.da_vocab = da_vocab
        self.encoder = UtteranceEncoder(utt_input_size=len(utt_vocab.word2id), embed_size=config['NRG']['UTT_EMBED'],
                               utterance_hidden=config['NRG']['UTT_HIDDEN'], padding_idx=utt_vocab.word2id['<PAD>'],
                               fine_tuning=False).cuda()
        self.decoder = UtteranceDecoder(utterance_hidden_size=config['NRG']['DEC_HIDDEN'], utt_embed_size=config['NRG']['UTT_EMBED'],
                                        utt_vocab_size=len(utt_vocab.word2id)).cuda()
        self.config = config
        self.reward_fn = reward_fn

    def forward(self, X, Y, X_da, turn, step_size):
        CE_loss = 0
        base_loss = 0
        encoder_hidden = self.encoder.initHidden(step_size)
        _, encoder_hidden = self.encoder(X, encoder_hidden)
        encoder_hidden = torch.cat((encoder_hidden[0, :, :], encoder_hidden[1, :, :]), dim=-1).unsqueeze(0)
        decoder_hidden = encoder_hidden
        # encoder_hidden: (1, batch_size, hidden_size)
        pred_seq = []
        base_seq = []
        prev_words = torch.tensor([[self.utt_vocab.word2id['<BOS>']] for _ in range(step_size)]).cuda()
        for j in range(Y.size(1)):
            logits, decoder_hidden, decoder_output = self.decoder(prev_words, decoder_hidden)
            filtered_logits = self.top_k_top_p_filtering(_logits=logits, top_k=self.config['NRG']['top_k'],
                                               top_p=self.config['NRG']['top_p'])
            probs = F.softmax(filtered_logits, dim=-1)
            _, base_topi = logits.topk(1)
            next_token = torch.multinomial(probs, 1).squeeze(-1)
            pred_seq.append(next_token)
            base_seq.append(base_topi.squeeze(-1))
            if self.config['RL']:
                CE_loss += self.criterion(probs, Y[:, j])
            else:
                base_loss += self.criterion(logits, Y[:, j])
            prev_words = Y[:, j].unsqueeze(-1)

        if self.config['RL']:
            pred_seq = torch.stack(pred_seq)
            base_seq = torch.stack(base_seq)
            pred_seq = [s for s in pred_seq.transpose(0,1).data.tolist()]
            base_seq = [s for s in base_seq.transpose(0,1).data.tolist()]
            ref_seq = [s for s in Y.data.tolist()]
            context = [s for s in X.data.tolist()]
            reward = self.reward_fn.reward(hyp=pred_seq, ref=ref_seq, context=context, da_context=X_da, turn=turn, step_size=step_size)
            b = self.reward_fn.reward(hyp=base_seq, ref=ref_seq, context=context, da_context=X_da, turn=turn, step_size=step_size)
            # print('sample: {}, base: {}'.format(reward, b))
            RL_loss = CE_loss * (reward - b)
            loss = CE_loss * self.config['NRG']['lambda'] + RL_loss * (1 - self.config['NRG']['lambda'])
            loss = loss.mean()
            reward = reward.mean().item()
        else:
            reward = 0
            loss = base_loss.mean()
            pred_seq = torch.stack(base_seq).transpose(0, 1).data.tolist()
        # torch.nn.utils.clip_grad_norm_(self.parameters(), self.config['clip'])

        if self.training:
            loss.backward()
        return loss.item(), reward, pred_seq

    def predict(self, X, step_size):
        with torch.no_grad():
            encoder_hidden = self.encoder.initHidden(step_size)
            _, encoder_hidden = self.encoder(X, encoder_hidden)
            encoder_hidden = torch.cat((encoder_hidden[0, :, :], encoder_hidden[1, :, :]), dim=-1).unsqueeze(0)
            # pred_seq, _ = self._beam_decode(decoder=self.decoder, decoder_hiddens=encoder_hidden, config=self.config)
            # pred_seq, _ = self._sample_decode(decoder_hidden=encoder_hidden, step_size=step_size)
            pred_seq, _ = self._greedy_decode(decoder_hidden=encoder_hidden, step_size=step_size)
        return torch.stack(pred_seq).transpose(0, 1).data.tolist()

    def perplexity(self, X, Y, step_size):
        X = X[0] if type(X) == list else X
        with torch.no_grad():
            loss = 0
            encoder_hidden = self.encoder.initHidden(step_size)
            _, encoder_hidden = self.encoder(X, encoder_hidden)
            encoder_hidden = torch.cat((encoder_hidden[0, :, :], encoder_hidden[1, :, :]), dim=-1).unsqueeze(0)
            decoder_hidden = encoder_hidden
            prev_words = torch.tensor([[self.utt_vocab.word2id['<BOS>']] for _ in range(step_size)]).cuda()
            for j in range(Y.size(1)):
                logits, decoder_hidden, decoder_output = self.decoder(prev_words, decoder_hidden)
                loss += self.criterion(logits, Y[:, j])
                prev_words = Y[:, j].unsqueeze(-1)
        return loss.data.tolist()

    def _greedy_decode(self, decoder_hidden, step_size):
        PAD_token = self.utt_vocab.word2id['<PAD>']
        prev_words = torch.tensor([[self.utt_vocab.word2id['<BOS>']] for _ in range(step_size)]).cuda()
        pred_seq = []
        for _ in range(self.config['max_len']):
            preds, decoder_hidden, _ = self.decoder(prev_words, decoder_hidden)
            _, topi = preds.topk(1)
            pred_seq.append(topi.squeeze(-1))
            prev_words = topi
            if all(ele == PAD_token for ele in prev_words):
                break
        return pred_seq, decoder_hidden

    def _sample_decode(self, decoder_hidden, step_size):
        PAD_token = self.utt_vocab.word2id['<PAD>']
        prev_words = torch.tensor([[self.utt_vocab.word2id['<BOS>']] for _ in range(step_size)]).cuda()
        pred_seq = []
        for _ in range(self.config['max_len']):
            logits, decoder_hidden, _ = self.decoder(prev_words, decoder_hidden)
            filtered_logits = self.top_k_top_p_filtering(_logits=logits, top_k=self.config['NRG']['top_k'], top_p=self.config['NRG']['top_p'])
            probs = F.softmax(filtered_logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            pred_seq.append(next_token.squeeze(-1))
            prev_words = next_token.clone()
            if all(ele == PAD_token for ele in next_token):
                break
        return pred_seq, decoder_hidden

    def top_k_top_p_filtering(self, _logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
        """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
            Args:
                logits: logits distribution shape (vocabulary size)
                top_k >0: keep only top k tokens with highest probability (top-k filtering).
                top_p >0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                    Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        """
        logits = _logits.clone()
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

        return decoded_batch, decoder_hidden

