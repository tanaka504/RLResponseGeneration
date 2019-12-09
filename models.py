from nn_blocks import *
from utils import *
import operator
from queue import PriorityQueue



class RL(nn.Module):
    def __init__(self, utt_vocab, utt_encoder, utt_context, utt_decoder, nli_model, ssn_model, config):
        super(RL, self).__init__()
        self.utt_vocab = utt_vocab
        self.utt_encoder = utt_encoder
        self.utt_context = utt_context
        self.utt_decoder = utt_decoder
        self.nli_model = nli_model
        self.ssn_model = ssn_model
        self.config = config

    def forward(self, X_utt, Y_utt, context, utt_context_hidden, step_size, criterion, last):
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

        # Decoding
        # TODO: not teacher forcing but inference
        pred_seq = []
        base_seq = []
        for j in range(len(Y_utt[0]) - 1):
            prev_words = Y_utt[:, j].unsqueeze(1)
            logits, decoder_hidden, _ = self.utt_decoder(prev_words, utt_dec_hidden)
            filtered_logits = self.top_k_top_p_filtering(_logits=logits, top_k=self.config['top_k'],
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
        # context = [s for s in X_utt.data.tolist()]
        reward = torch.tensor(self.reward(pred_seq, ref_seq, context)).cuda()
        b = torch.tensor(self.reward(base_seq, ref_seq, context)).cuda()
        print('sample: {}, base: {}'.format(reward, b))

        # Optimized with REINFORCE
        RL_loss = CE_loss * (reward - b)
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
        # TODO: Implement reward functions
        r_bleu = calc_bleu(reference, hypothesis)
        return r_bleu

    
    def predict(self, X_utt, utt_context_hidden, step_size):
        with torch.no_grad():
            utt_decoder_hidden = self._encoding(X_utt=X_utt, utt_context_hidden=utt_context_hidden, step_size=step_size)
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
        utt_encoder_output = utt_encoder_output.unsqueeze(1)
        # Update Context Encoder
        utt_context_output, utt_context_hidden = self.utt_context(utt_encoder_output, utt_context_hidden) # (batch_size, 1, UTT_CONTEXT)

        return utt_context_hidden

    def _greedy_decode(self, prev_words, decoder, decoder_hidden):
        PAD_token = self.utt_vocab.word2id['<PAD>']
        pred_seq = []
        for _ in range(self.config['max_len']):
            preds, decoder_hidden, _ = decoder(prev_words, decoder_hidden)
            _, topi = preds.topk(1)
            pred_seq.append(topi)
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
                criterion, last, context=None):
        loss = 0

        # Encoding
        utt_dec_hidden = self._encoding(X_utt=X_utt, utt_context_hidden=utt_context_hidden, step_size=step_size)

        # Response Decode
        pred_seq = []
        for j in range(len(Y_utt[0]) - 1):
            prev_words = Y_utt[:, j].unsqueeze(1)
            preds, utt_decoder_hidden, utt_decoder_output = self.utt_decoder(prev_words, utt_dec_hidden)
            _, topi = preds.topk(1)
            pred_seq.append(topi.squeeze(-1))
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
            return loss.item(), utt_context_hidden, 0, torch.stack(pred_seq).transpose(0, 1).data.tolist()

    def predict(self, X_utt, utt_context_hidden, step_size=1):
        with torch.no_grad():
            utt_dec_hidden = self._encoding(X_utt=X_utt, utt_context_hidden=utt_context_hidden, step_size=step_size)

            utt_decoder_hidden = utt_dec_hidden
            prev_words = torch.tensor([[self.utt_vocab.word2id['<BOS>']] for _ in range(step_size)]).cuda()

            if self.config['beam_size']:
                pred_seq, utt_decoder_hidden = self._beam_decode(decoder=self.utt_decoder, decoder_hiddens=utt_decoder_hidden, config=self.config)
                pred_seq = [seq[0] for seq in pred_seq]
            else:
                # pred_seq, utt_decoder_hidden = self._greedy_decode(prev_words, self.utt_decoder, utt_decoder_hidden, config=self.config)
                # pred_seq = torch.stack(pred_seq).permute(1, 0).data.tolist()
                pred_seq, utt_decoder_hidden = self._sample_decode(prev_words=prev_words, decoder=self.utt_decoder, decoder_hidden=utt_decoder_hidden)
                pred_seq = torch.stack(pred_seq)
                pred_seq = [s for s in pred_seq.transpose(0, 1).data.tolist()]
        return pred_seq, utt_context_hidden

    def _encoding(self, X_utt, utt_context_hidden, step_size):
        # Encode Utterance
        utt_encoder_hidden = self.utt_encoder.initHidden(step_size)
        utt_encoder_output, utt_encoder_hidden = self.utt_encoder(X_utt, utt_encoder_hidden)  # (batch_size, 1, UTT_HIDDEN)
        utt_encoder_output = utt_encoder_output.unsqueeze(1)
        # Update Context Encoder
        utt_context_output, utt_context_hidden = self.utt_context(utt_encoder_output, utt_context_hidden) # (batch_size, 1, UTT_HIDDEN)

        return utt_context_hidden

    def _greedy_decode(self, prev_words, decoder, decoder_hidden):
        PAD_token = self.utt_vocab.word2id['<PAD>']
        pred_seq = []
        for _ in range(self.config['max_len']):
            preds, decoder_hidden, _ = decoder(prev_words, decoder_hidden)
            _, topi = preds.topk(1)
            pred_seq.append(topi)
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

    def calc_bleu(self, refs, hyps):
        return corpus_bleu([[ref] for ref in refs], hyps)


class seq2seq(nn.Module):
    def __init__(self, encoder, decoder, criterion, src_vocab, tgt_vocab, config):
        super(seq2seq, self).__init__()
        self.criterion = criterion
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.encoder = encoder
        self.decoder = decoder
        self.config = config

    def forward(self, X, Y, step_size):
        CE_loss = 0
        base_loss = 0
        encoder_hidden = self.encoder.initHidden(step_size)
        encoder_output, encoder_hidden = self.encoder(X, encoder_hidden)
        encoder_hidden = torch.cat((encoder_hidden[0, :, :], encoder_hidden[1, :, :]), dim=-1).unsqueeze(0)
        pred_seq = []
        base_seq = []
        for j in range(len(Y[0]) - 1):
            prev_words = Y[:, j].unsqueeze(1)
            logits, decoder_hidden, decoder_output = self.decoder(prev_words, encoder_hidden)
            filtered_logits = self.top_k_top_p_filtering(_logits=logits, top_k=self.config['top_k'],
                                               top_p=self.config['top_p'])
            probs = F.softmax(filtered_logits, dim=-1)
            _, base_topi = logits.topk(1)
            next_token = torch.multinomial(probs, 1).squeeze(-1)
            pred_seq.append(next_token)
            base_seq.append(base_topi)
            if self.config['RL']:
                CE_loss += self.criterion(probs.view(-1, len(self.tgt_vocab.word2id)), Y[:, j+1])
            else:
                base_loss += self.criterion(logits.view(-1, len(self.tgt_vocab.word2id)), Y[:, j+1])

        if self.config['RL']:
            pred_seq = torch.stack(pred_seq)
            base_seq = torch.stack(base_seq)
            pred_seq = [s for s in pred_seq.transpose(0,1).data.tolist()]
            base_seq = [[w[0] for w in s] for s in base_seq.transpose(0,1).data.tolist()]
            ref_seq = [s for s in Y.data.tolist()]
            context = [s for s in X.data.tolist()]
            reward = self.reward(hyp=pred_seq, ref=ref_seq, context=context)
            b = self.reward(hyp=base_seq, ref=ref_seq, context=context)
            RL_loss = CE_loss * (reward - b)
            loss = CE_loss * self.config['lambda'] + RL_loss * (1 - self.config['lambda'])
            loss = loss.mean()
        else:
            reward = 0
            loss = base_loss.mean()
        torch.nn.utils.clip_grad_norm_(self.parameters(), self.config['clip'])

        if self.training:
            loss.backward()
        return loss.item(), reward

    def predict(self, X, step_size):
        with torch.no_grad():
            encoder_hidden = self.encoder.initHidden(step_size)
            encoder_output, encoder_hidden = self.encoder(X, encoder_hidden)
            encoder_hidden = torch.cat((encoder_hidden[0, :, :], encoder_hidden[1, :, :]), dim=-1).unsqueeze(0)
            pred_seq, _ = self._beam_decode(decoder=self.decoder, decoder_hiddens=encoder_hidden, config=self.config)
        return pred_seq

    def reward(self, hyp, ref, context):
        r_bleu = calc_bleu(ref, hyp)
        return r_bleu

    def _greedy_decode(self, prev_words, decoder, decoder_hidden):
        EOS_token = self.tgt_vocab.word2id['<EOS>']
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
        EOS_token = self.tgt_vocab.word2id['<EOS>']
        pred_seq = []
        for _ in range(self.config['max_len']):
            logits, decoder_hidden, _ = decoder(prev_words, decoder_hidden)
            filtered_logits = self.top_k_top_p_filtering(_logits=logits, top_k=self.config['top_k'], top_p=self.config['top_p'])
            probs = F.softmax(filtered_logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            pred_seq.append(next_token)
            prev_words = torch.tensor(next_token).cuda()
            if next_token == EOS_token:
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
        BOS_token = self.tgt_vocab.word2id['<BOS>']
        EOS_token = self.tgt_vocab.word2id['<EOS>']
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

