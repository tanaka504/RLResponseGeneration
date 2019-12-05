import os, re, json, sys

file_pattern = re.compile(r'^sw\_([0-9]+?)\_([0-9]+?)\.utt\.txt$')
line_pattern = re.compile(r'^(.*?)\t(.*?)\t(.*?)$')

tmp_da = {'A': None, 'B': None}

def preprocess(dir_path, filename):
    with open(os.path.join(dir_path, filename), 'r') as f, \
        open(os.path.join('./data/corpus/', filename[:-7] + 'jsonlines'), 'w') as out_f:
        data = f.read().split('\n')
        prev_caller = None
        das = []
        sentences = []

        for line in data:
            m = line_pattern.search(line)
            if not m is None:
                current_caller = m.group(1)
                if m.group(2) == '+':
                    da = tmp_da[current_caller]
                else:
                    da = m.group(2)
                    tmp_da[current_caller] = da
                assert da is not None, filename
                if current_caller == prev_caller:
                    das.append(da)
                    sentences.append(m.group(3))
                else:
                    if len(das) > 0 and len(sentences) > 0:
                        out_f.write(json.dumps({'caller': prev_caller,
                                                'DA': das,
                                                'sentence': sentences}))
                        out_f.write('\n')
                    das = [da]
                    sentences = [m.group(3)]
                    prev_caller = current_caller

def FileIter():
    for i in range(14):
        dir_path = os.path.join('./data/swda', 'sw{:02}utt'.format(i))
        print('preprocessing in {}'.format(dir_path))
        files = [f for f in os.listdir(dir_path) if file_pattern.match(f)]
        for i, filename in enumerate(files, 1):
            preprocess(dir_path, filename)
            print('\rFinish preprocess {}/{} files'.format(i, len(files)), end='')
        print()

def DNLIPreprocess(prefix):
    jsondata = json.load(open('./data/corpus/dnli/dialogue_nli/dialogue_nli_{}.jsonl'.format(prefix)))
    f = open('./data/corpus/dnli/dialogue_nli_{}.tsv'.format(prefix), 'w')
    f.write('\n'.join(['{}\t{}\t{}'.format(line['sentence1'], line['sentence2'], line['label']) for line in jsondata]))
    f.close()

def RITEPreprocess(prefix):
    jsondata = json.load(open('./data/corpus/RITE/RITE2_JA_{}_mc.json'.format(prefix)))
    f = open('./data/corpus/RITE/RITE_{}.tsv'.format(prefix), 'w')
    f.write('\n'.join(['{}\t{}\t{}'.format(line['t1'], line['t2'], line['label']) for line in jsondata]))
    f.close()

def MTPreprocess():
    from NLI import JumanTokenizer
    from nltk import tokenize
    src_tokenizer = JumanTokenizer()
    pattern = re.compile('^A: (.*?)#ID=(.*?)$')
    f = open('./data/corpus/mt.tsv', 'w')
    for idx, line in enumerate(open('./data/corpus/mt.corpus', 'r').readlines()):
        m = pattern.search(line)
        if m is None: continue
        x, y = m.group(1).split('\t')
        x = ' '.join(src_tokenizer.tokenize(x))
        y = ' '.join(tokenize.word_tokenize(y.lower()))
        f.write(x + '\t' + y + '\n')
    f.close()

if __name__ == '__main__':
    MTPreprocess()