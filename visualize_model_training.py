import pickle
import matplotlib.pyplot as plt


def visualize_model_training(gru_spec_file, short_gru_spec_file, lstm_spec_file, short_lstm_spec_file, gru_mel_file, lstm_mel_file, gru_mfcc_file, lstm_mfcc_file):
    gru_spec_file = open(gru_spec_file, 'rb')
    short_gru_spec_file = open(short_gru_spec_file, 'rb')
    lstm_spec_file = open(lstm_spec_file, 'rb')
    short_lstm_spec_file = open(short_lstm_spec_file, 'rb')
    gru_mel_file = open(gru_mel_file, 'rb')
    lstm_mel_file = open(lstm_mel_file, 'rb')
    gru_mfcc_file = open(gru_mfcc_file, 'rb')
    lstm_mfcc_file = open(lstm_mfcc_file, 'rb')

    history_1 = pickle.load(gru_spec_file)
    history_2 = pickle.load(short_gru_spec_file)
    lstm_spec_history = pickle.load(lstm_spec_file)
    short_lstm_spec_history = pickle.load(short_lstm_spec_file)
    gru_mel_history = pickle.load(gru_mel_file)
    lstm_mel_history = pickle.load(lstm_mel_file)
    gru_mfcc_history = pickle.load(gru_mfcc_file)
    lstm_mfcc_history = pickle.load(lstm_mfcc_file)

    plt.plot(history_1['loss'])
    plt.plot(history_2['loss'])
    plt.plot(lstm_spec_history['loss'])
    plt.plot(short_lstm_spec_history['loss'])
    plt.plot(gru_mel_history['loss'])
    plt.plot(lstm_mel_history['loss'])
    plt.plot(gru_mfcc_history['loss'])
    plt.plot(lstm_mfcc_history['loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.yscale('log')
    plt.legend(['GRU_spec_20400 loss', 'GRU_spec_5712 loss', 'LSTM_spec_20400 loss', 'LSTM_spec_5712 loss', 'GRU_mel loss', 'LSTM_mel loss', 'GRU MFCC loss','LSTM MFCC loss'], loc='upper right')
    plt.savefig('model_histories/all_training_graphs.png', dpi=300)
    plt.show()

visualize_model_training('model_histories/GRU_SPEC_history','model_histories/short_GRU_SPEC_history','model_histories/LSTM_SPEC_history', 'model_histories/short_LSTM_SPEC_history', 'model_histories/GRU_MEL_history', 'model_histories/LSTM_MEL_history', 'model_histories/GRU_MFCC_history','model_histories/LSTM_MFCC_history')