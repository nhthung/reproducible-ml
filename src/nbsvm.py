import numpy as np
from keras.models import Model
from keras.layers.core import Activation
from keras.layers import Input, Embedding, Flatten, dot
from keras.optimizers import Adam

def build_nbsvm_data(dtm_train, labels, max_len=500):
    x = []
    nwds = []
    for row in dtm_train:
        indices = (row.indices + 1).astype(np.int64)
        np.append(nwds, len(indices))
        data = (row.data).astype(np.int64)
        count_dict = dict(zip(indices, data))
        seq = [k for k, v in count_dict.items() for i in range(v)]
        seq_len = len(seq)
        nwds.append(seq_len)
        seq = np.pad(seq, (max_len-seq_len, 0), mode='constant') if seq_len < max_len else seq[-max_len:]
        x.append(seq)
    nwds = np.array(nwds)
    print(f'Sequence stats: Avg: {nwds.mean():.2f} | Max: {nwds.max()} | Min: {nwds.min()}')
    x_train = np.array(x)

    def nb_prob(dtm, y, y_i):
        return (dtm[y==y_i].sum(0)+1) / ((y==y_i).sum()+2)

    nb_ratios = np.log(nb_prob(dtm_train, labels, 1) / nb_prob(dtm_train, labels, 0))
    nb_ratios = np.squeeze(np.array(nb_ratios))

    return x_train, nb_ratios

def get_nbsvm_model(num_words, max_len=500, nb_ratios=None):
    embedding_matrix = np.zeros((num_words, 1))
    for i in range(1, num_words):
        embedding_matrix[i] = nb_ratios[i-1] if nb_ratios is not None else 1

    inp = Input(shape=(max_len,))
    r = Embedding(num_words, 1, input_length=max_len, weights=[embedding_matrix], trainable=False)(inp)
    x = Embedding(num_words, 1, input_length=max_len, embeddings_initializer='glorot_normal')(inp)
    x = dot([r,x], axes=1)
    x = Flatten()(x)
    x = Activation('sigmoid')(x)
    model = Model(inputs=inp, outputs=x)
    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])
    return model
