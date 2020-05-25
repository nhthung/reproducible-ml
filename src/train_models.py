import os
import time
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from keras.callbacks import EarlyStopping
from keras.utils import plot_model

from data import load_mr, load_sst2
from nbsvm import get_nbsvm_model
from pipeline import linear_svc_pipeline, nbsvm_pipeline

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
PLOT_SVC_MODEL = False
PLOT_NBSVM_MODEL = False

def execute_pipeline(pipeline, x, y, x_v, y_v, x_t, y_t):
    start = time.time()
    pipeline.fit(x, y)
    train_time = time.time()
    print(f'Training time: {train_time-start}')
    print(f'Training accuracy: {pipeline.score(x, y)}')
    print(f'Validation accuracy: {pipeline.score(x_v, y_v)}')
    print(f'Test accuracy: {pipeline.score(x_t, y_t)}')
    print(f'Scoring time: {time.time()-train_time}s')

def plot_model_history(history, model_name, dataset_name):
    history_dict = history.history
    acc = history_dict['acc']
    val_acc = history_dict['val_acc']
    epochs = range(1, len(acc) + 1)
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title(f'Training and Validation Accuracy of {model_name} - {dataset_name}')
    plt.xlabel('Epochs'), plt.ylabel('Accuracy'), plt.legend()
    plt.show()

if __name__ == '__main__':
    print('Loading data...')
    start = time.time()
    mr_data, mr_labels = load_mr()
    sst_train_data, sst_train_labels, sst_dev_data, sst_dev_labels, sst_test_data, sst_test_labels = load_sst2()
    print(f'Time to load data: {time.time()-start}s')

    ########################################
    print('##### Training Linear SVC #####')
    ########################################

    print('Linear SVC - SST2 Dataset')
    ##################################
    pipeline = linear_svc_pipeline(max_features=None, ngram=2, tfidf=True)
    execute_pipeline(pipeline, sst_train_data, sst_train_labels, sst_dev_data, sst_dev_labels, sst_test_data, sst_test_labels)

    print('Linear SVC - MR Dataset')
    ################################
    pipeline = linear_svc_pipeline(max_features=None, ngram=2, tfidf=True)
    scores = cross_val_score(pipeline, mr_data, mr_labels, cv=10, n_jobs=-1)
    print(f'Training Accuracy: {scores.mean()}')
    mr_pred = cross_val_predict(pipeline, mr_data, mr_labels, cv=10, n_jobs=-1)
    print(f'Validation Accuracy: {accuracy_score(mr_labels, mr_pred)}')


    #############################################
    print('##### Training Naive Bayes SVM #####')
    #############################################
    early_stop = EarlyStopping(monitor='val_acc', patience=4, verbose=1)

    print('Naive Bayes SVM - MR Dataset')
    #####################################
    hp_grid = {
        'ngram': [2, 3],
        'max_features': [5000, 15000, 50000],
        'batch_size': [16, 32, 64],
        'epochs': [5, 10, 15]
    }

    hp_scores = []
    hp_histories = []
    for ngram in hp_grid['ngram']:
        x_mr, nb_ratios, num_words = nbsvm_pipeline(mr_data, mr_labels, max_features=100000, ngram=ngram)

        for epochs in hp_grid['epochs']:
            for batch_size in hp_grid['batch_size']:
                print(f'Hyper Parameters: Ngram={ngram} | Batch Size={batch_size} | Epochs={epochs}')

                cv_scores = []
                cv_histories = []
                kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
                for trn, tst in kfold.split(mr_data, mr_labels):
                    model = get_nbsvm_model(num_words, nb_ratios=nb_ratios)
                    history = model.fit(x_mr[trn], mr_labels[trn], validation_data=(x_mr[tst], mr_labels[tst]), batch_size=batch_size, epochs=epochs, callbacks=[early_stop], verbose=0)
                    score = model.evaluate(x_mr[tst], mr_labels[tst], verbose=0)[1]

                    cv_scores.append(score)
                    cv_histories.append(history)
                    print(f'Val Acc: {score:.5f}')

                best_model_idx = cv_scores.index(max(cv_scores))
                hp_scores.append(cv_scores[best_model_idx])
                hp_histories.append(cv_histories[best_model_idx])
                print(f'Avg: {np.mean(cv_scores):.5f} (+/- {np.std(cv_scores):.5f})')

    best_model = hp_histories[hp_scores.index(max(hp_scores))]
    plot_model_history(best_model, 'Naive Bayes SVM', 'MR')

    print('Naive Bayes SVM - SST2 Dataset')
    #######################################
    hp_grid = {
        'ngram': [2, 3],
        'max_features': [5000, 15000, 50000],
        'batch_size': [16, 32, 64],
        'epochs': [5, 10, 15]
    }
    sst_train_len, sst_dev_len, sst_test_len = map(len, [sst_train_data, sst_dev_data, sst_test_data])
    split1, split2, split3 = sst_train_len, sst_train_len + sst_dev_len, sst_train_len + sst_dev_len + sst_test_len

    sst_data = np.concatenate([sst_train_data, sst_dev_data, sst_test_data])
    sst_labels = np.concatenate([sst_train_labels, sst_dev_labels, sst_test_labels])

    hp_scores = []
    hp_histories = []
    for ngram in hp_grid['ngram']:
        x_sst, nb_ratios, num_words = nbsvm_pipeline(sst_data, sst_labels, max_features=100000, ngram=ngram)
        x_train, x_dev, x_test = x_sst[:split1], x_sst[split1:split2], x_sst[split2:split3]
        y_train, y_dev, y_test = sst_labels[:split1], sst_labels[split1:split2], sst_labels[split2:split3]

        for epochs in hp_grid['epochs']:
            for batch_size in hp_grid['batch_size']:
                print(f'Hyper Parameters: Ngram={ngram} | Batch Size={batch_size} | Epochs={epochs}')

                model = get_nbsvm_model(num_words, nb_ratios=nb_ratios)
                history = model.fit(x_train, y_train, validation_data=(x_dev, y_dev), batch_size=batch_size, epochs=epochs, callbacks=[early_stop], verbose=0)
                score = model.evaluate(x_test, y_test, verbose=0)[1]

                hp_scores.append(score)
                hp_histories.append(history)
                print(f'Test Acc: {score:.5f}')

    best_model = hp_histories[hp_scores.index(max(hp_scores))]
    plot_model_history(best_model, 'Naive Bayes SVM', 'SST2')

    # Plot NBSVM model architecture
    if PLOT_NBSVM_MODEL:
        plot_model(model, to_file='./project4/figures/nbsvm_arch.png', show_shapes=True)
