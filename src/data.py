import re
import os.path as op
import numpy as np
from pandas import read_pickle

DATA_DIR = op.abspath(op.join(__file__, op.pardir, op.pardir, 'data'))
MR_FILE = 'MR.pkl'
SST2_FILE = 'SST2.pkl'

def load_data(filename):
    df = read_pickle(op.join(DATA_DIR, filename))
    train_data = []
    train_labels = []
    dev_data = []
    dev_labels = []
    test_data = []
    test_labels = []
    for _, row in df.itertuples():
        split = row['split']
        sentence = clean_sentence(row['sentence'])
        label = row['label']

        if split == 'train':
            train_data.append(sentence)
            train_labels.append(label)
        elif split == 'dev':
            dev_data.append(sentence)
            dev_labels.append(label)
        elif split == 'test':
            test_data.append(sentence)
            test_labels.append(label)
        else:
            raise Exception(f'Unknown label: {row["split"]}')
    if len(dev_data) == 0 or len(test_data) == 0:
        return np.array(train_data), np.array(train_labels)
    else:
        return np.array(train_data), np.array(train_labels), np.array(dev_data), np.array(dev_labels), np.array(test_data), np.array(test_labels)

def load_mr():
    return load_data(MR_FILE)

def load_sst2():
    return load_data(SST2_FILE)

def clean_sentence(sentence):
    replacements = {
        "'m": ' am', "'s": ' is', "'t": ' not', "'d": ' would',
        "'re": ' are', "'ll": ' will', "'ve": ' have'
    }

    regexp = re.compile('|'.join(map(re.escape, replacements)))
    return regexp.sub(lambda match: replacements[match.group(0)], sentence)
