import torch
import torch.nn as nn
from torch import optim
from train import initialize_env, parse
import time, random
from utils import *
from sklearn.metrics import accuracy_score
from pyknp import Juman
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from transformers import glue_convert_examples_to_features as convert_examples_to_features
from transformers import glue_compute_metrics as compute_metrics
from tqdm import tqdm
from transformers import BertConfig, BertForSequenceClassification, BertTokenizer
from transformers.data.processors.utils import InputExample
from NLI_processor import processors, output_modes


class NLI:
    def __init__(self):
        self.model = BertForSequenceClassification.from_pretrained('./data/model_en/bert_fine_tuning').cuda()
        self.tokenizer = BertTokenizer.from_pretrained('./data/model_en/bert_fine_tuning')

    def predict(self, x1, x2):
        """
        param x1: batch of sentence1 (batch_size, seq_len)
        param x2: batch of sentence2 (batch_size, seq_len)
        """
        output_dir = './data/model_en/bert_fine_tuning'
        # Loop to handle MNLI double evaluation (matched, mis-matched)
        eval_task_names = ("dnli",)
        eval_outputs_dirs = (output_dir,)

        for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
            eval_dataset = self.load_and_cache_examples(x1, x2, eval_task, self.tokenizer)

            if not os.path.exists(eval_output_dir):
                os.makedirs(eval_output_dir)

            # Note that DistributedSampler samples randomly
            eval_sampler = SequentialSampler(eval_dataset)
            eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=32)

            eval_loss = 0.0
            nb_eval_steps = 0
            preds = None
            out_label_ids = None
            for batch in tqdm(eval_dataloader, desc="Evaluating"):
                self.model.eval()
                batch = tuple(t.cuda() for t in batch)

                with torch.no_grad():
                    inputs = {'input_ids': batch[0],
                              'attention_mask': batch[1],
                              'labels': batch[3]}
                    inputs['token_type_ids'] = batch[2]
                    outputs = self.model(**inputs)
                    tmp_eval_loss, logits = outputs[:2]

                    eval_loss += tmp_eval_loss.mean().item()
                nb_eval_steps += 1
                if preds is None:
                    preds = logits.detach().cpu().numpy()
                    out_label_ids = inputs['labels'].detach().cpu().numpy()
                else:
                    preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                    out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)
        return preds

    def load_and_cache_examples(self, x1, x2, task, tokenizer):
        processor = processors[task]()
        output_mode = output_modes[task]
        # Load data features from cache or dataset file
        label_list = processor.get_labels()
        examples = []
        for t1, t2 in zip(x1, x2):
            guid = "%s-%s" % ('dev_matched', t1)
            examples.append(InputExample(guid=guid, text_a=t1, text_b=t2, label='negative'))

        features = convert_examples_to_features(examples,
                                                tokenizer,
                                                label_list=label_list,
                                                max_length=128,
                                                output_mode=output_mode,
                                                pad_on_left=False,
                                                # pad on the left for xlnet
                                                pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                                                pad_token_segment_id=0,
                                                )

        # Convert to Tensors and build dataset
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
        if output_mode == "classification":
            all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
        elif output_mode == "regression":
            all_labels = torch.tensor([f.label for f in features], dtype=torch.float)

        dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
        return dataset

class JumanTokenizer:
    def __init__(self):
        self.juman = Juman()

    def tokenize(self, text):
        text.translate(str.maketrans({chr(0x0021 + i): chr(0xFF01 + i) for i in range(94)}))
        text = re.sub(r'\s', '　', text)
        result = self.juman.analysis(text)
        return [mrph.midasi for mrph in result.mrph_list()]

def main():
    label2id = {'positive': 0, 'negative': 1, 'neutral': 2}
    predictor = NLI()
    f = open('./data/corpus/dnli/dialogue_nli_test.tsv', 'r')
    testdata = [line.strip().split('\t') for line in f.readlines()]
    x1, x2, label = zip(*testdata)
    preds = predictor.predict(x1[:1000], x2[:1000])
    preds = np.argmax(preds, axis=1)
    print(accuracy_score(y_pred=preds, y_true=[label2id[t] for t in label[:1000]]))

if __name__ == '__main__':
    args = parse()
    main()

