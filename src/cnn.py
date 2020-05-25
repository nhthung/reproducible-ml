import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import Conv1D, MaxPooling1D, Embedding, Dropout, Dense, concatenate, Flatten, Input
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import plot_model
from matplotlib import pyplot as plt

from data import load_mr, load_sst2

def get_conv_layers(x_input, n_grams=(3, 4, 5), feature_maps=100):
    branches = []
    for n in n_grams:
        branch = Conv1D(filters=feature_maps, kernel_size=n, activation='relu')(x_input)
        branch = MaxPooling1D(pool_size=2, strides=None, padding='valid')(branch)
        branch = Flatten()(branch)
        branches.append(branch)
    return branches

mr_data, mr_labels = load_mr()
sst_train_data, sst_train_labels, sst_dev_data, sst_dev_labels, sst_test_data, sst_test_labels = load_sst2()

mr_data_train, mr_data_test, mr_labels_train, mr_labels_test = train_test_split(mr_data, mr_labels, test_size=0.1, random_state=42)

tokenizer = Tokenizer(num_words=None)
tokenizer.fit_on_texts(mr_data_train)

X_train = tokenizer.texts_to_sequences(mr_data_train)
X_test = tokenizer.texts_to_sequences(mr_data_test)

num_words = len(tokenizer.word_index) + 1

max_len = 50
X_train = pad_sequences(X_train, padding='post', maxlen=max_len)
X_test = pad_sequences(X_test, padding='post', maxlen=max_len)

embed_dim = 100
embeddings = np.random.rand(num_words, embed_dim)
embed_layer = Embedding(input_dim=embeddings.shape[0], output_dim=embeddings.shape[1], input_length=max_len, weights=[embeddings], trainable=True)

i = Input(shape=(max_len,))
embed = embed_layer(i)
conv_layers = get_conv_layers(embed)
concat = concatenate(conv_layers)
concat = Dropout(0.5)(concat)
fc = Dense(1, activation='sigmoid')(concat)

model = Model(inputs=i, outputs=fc)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()
plot_model(model, to_file='./project4/cnn_arch.png', show_shapes=True)

history = model.fit(X_train, mr_labels_train, epochs=10, verbose=1, validation_data=(X_test, mr_labels_test), batch_size=64)
loss, accuracy = model.evaluate(X_train, mr_labels_train)
print(f'Training accuracy: {accuracy}')
loss, accuracy = model.evaluate(X_test, mr_labels_test)
print(f'Validation accuracy: {accuracy}')

# X_test = tokenizer.texts_to_sequences(sst_test_data)
# X_test = pad_sequences(X_test, padding='post', maxlen=max_len)
# loss, accuracy = model.evaluate(X_test, sst_test_labels)
# print(f'Test accuracy: {accuracy}')

history_dict = history.history
acc = history_dict['acc']
val_acc = history_dict['val_acc']
epcs = range(1, len(acc) + 1)
plt.plot(epcs, acc, 'bo', label='Training acc')
plt.plot(epcs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
