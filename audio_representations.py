import numpy, pandas, librosa, os, sklearn.preprocessing, joblib, glob, pickle
from keras.models import model_from_json
import re
numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

def file_to_spectrogram(filename, n_fft, hop_length):
    y,sr = librosa.load(filename)
    spectrogram = numpy.abs(librosa.core.stft(y, n_fft=n_fft, hop_length=hop_length))
    return spectrogram

def convert_files_to_specs(directory, n_fft, hop_length):
    all_songs = pandas.read_csv(directory, header=None, sep=';', index_col=False, names=['artist', 'title', 'lyrics', 'link', 'path'])
    scaler = sklearn.preprocessing.MinMaxScaler()
    # specs = numpy.empty(shape=[16594,2206,408])
    for i, song in all_songs.iterrows():
        if i >= 12247:
            filename = song['path'].split('/')
            wav_file = '/Users/m_vys/PycharmProjects/cleaned_wav_files/' + filename[5][:-3] + 'wav'
            spectrogram = file_to_spectrogram(wav_file, n_fft, hop_length)
            spectrogram = scaler.fit_transform(spectrogram)
            numpy.save('/Users/m_vys/PycharmProjects/similarity_and_evaluation/spectrograms/' + str(i) + '_' + str(song['artist'] + ' - ' + str(song['title'])), spectrogram)
            print(i)



# convert_files_to_specs('not_empty_songs', 4410, 812)


def file_to_mel(filename, nmels, n_fft, hop_length):
    y,sr = librosa.load(filename)
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=320, n_fft=4410, hop_length=812)
    return mel_spectrogram


def convert_files_to_mels(directory, n_mels, n_fft, hop_length):
    all_songs = pandas.read_csv(directory, header=None, sep=';', index_col=False, names=['artist', 'title', 'lyrics', 'link', 'path'])
    scaler = sklearn.preprocessing.MinMaxScaler()
    mels = numpy.empty(shape=[16594, 320, 408])
    for i, song in all_songs.iterrows():
        filename = song['path'].split('/')
        wav_file = '/Users/m_vys/PycharmProjects/cleaned_wav_files/' + filename[5][:-3] + 'wav'
        mel_spectrogram = file_to_mel(wav_file, n_mels, n_fft, hop_length)
        mel_spectrogram = scaler.fit_transform(mel_spectrogram)
        mels[i] = mel_spectrogram
        print(i)

    numpy.save('song_mel_spectrograms', mels)

def convert_files_to_mfcc(directory, n_mfcc):
    all_songs = pandas.read_csv(directory, header=None, sep=';', index_col=False,
                                names=['artist', 'title', 'lyrics', 'link', 'path'])
    scaler = sklearn.preprocessing.MinMaxScaler()
    mfcc_representations = numpy.empty([16594, 128, 646])
    for i, song in all_songs.iterrows():
        filename = song['path'].split('/')
        wav_file = '/Users/m_vys/PycharmProjects/cleaned_wav_files/' + filename[5][:-3] + 'wav'
        vector = librosa.load(wav_file)
        mfcc_representation = librosa.feature.mfcc(vector[0], vector[1], n_mfcc= n_mfcc)
        mfcc_representation = scaler.fit_transform(mfcc_representation)
        mfcc_representations[i] = mfcc_representation
        print(i)

    numpy.save('mfcc_representations', mfcc_representations)


def get_PCA_Mel_representations(model, mel_spec_matrix, repr_name):
    model = joblib.load(model)
    scaler = sklearn.preprocessing.MinMaxScaler()
    mel_specs = numpy.load(mel_spec_matrix)
    mel_specs = scaler.fit_transform(mel_specs.reshape([16594, 130560]))
    pca_mel_representations = numpy.empty([16594, 5715])
    for i in range(16594):
        mel_spec = mel_specs[i]
        pca_mel_spec = model.transform(mel_spec.reshape(1,-1))
        pca_mel_representations[i] = pca_mel_spec
        print(i)

    numpy.save(repr_name, pca_mel_representations)

def get_MFCC_representations(mfcc_model, mfcc_weights, mfcc_representations, repr_name):
    new_representations = numpy.empty([16594, 5168])
    mfcc_representations = numpy.load(mfcc_representations).reshape([16594, 646, 128])
    json_file = open(mfcc_model, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    MFCC_model = model_from_json(loaded_model_json)
    # load weights into new model
    MFCC_model.load_weights(mfcc_weights)
    print("Loaded model from disk")
    for i in range(16594):
        neural_mfcc = MFCC_model.predict(mfcc_representations[i].reshape(1, 646, 128))
        new_representations[i] = neural_mfcc.reshape(1, 5168)
        print(i)

    numpy.save(repr_name, new_representations)


def get_PCA_Spec_representations(pca_model, directory, repr_name):
    representations = numpy.empty([16594, 1106])
    pca_model = joblib.load(pca_model)
    i = 0
    for file in sorted(glob.glob(directory + '/*.npy'), key=numericalSort):
        spec = numpy.load(file).reshape([1,900048])
        pca_spec = pca_model.transform(spec)
        print(i, pca_spec.shape)
        representations[i] = pca_spec
        i = i+1
    numpy.save(repr_name, representations)

def get_PCA_Tf_idf_representations(model, tf_idf_matrix, repr_name):
    model = joblib.load(model)
    scaler = sklearn.preprocessing.MinMaxScaler()
    mel_specs = numpy.load(tf_idf_matrix)
    mel_specs = scaler.fit_transform(mel_specs.reshape([16594, 40165]))
    pca_mel_representations = numpy.empty([16594, 4457])
    for i in range(16594):
        mel_spec = mel_specs[i]
        pca_mel_spec = model.transform(mel_spec.reshape(1, -1))
        pca_mel_representations[i] = pca_mel_spec
        print(i)

    numpy.save(repr_name, pca_mel_representations)



def get_tf_idf_representations(tf_idf_model, songs, tf_idf_numpy_filename):
    vectorizer = pickle.load(open(tf_idf_model, 'rb'))

# get_PCA_Spec_representations('/mnt/0/big_pca_model', 'mnt/0/spectrograms', 'mnt/0/pca_spec_representations_1106')
# get_PCA_Mel_representations('mnt/0/mel_spec_pca_model_90_ratio', 'mnt/0/song_mel_spectrograms.npy', 'mnt/0//mel_spec_representations_5717')
#convert_files_to_mels('not_empty_songs', 329, 4410, 812)
# convert_files_to_mfcc('not_empty_songs', 320)

# get_MFCC_representations('/mnt/0/models/GRU_MFCC_model.json', '/mnt/0/GRU_MFCC_model.h5', '/mnt/0/mfcc_representations.npy', '/mnt/0/gru_mfcc_representations')
# print('gru_mfcc representations saved')
# get_MFCC_representations('/mnt/0/LSTM_MFCC_model.json', '/mnt/0/LSTM_MFCC_model.h5', '/mnt/0/mfcc_representations.npy', '/mnt/0/lstm_mfcc_representations')
# print('lstm mfcc representations saved')

# from Distances import save_neural_network
#
# save_neural_network('mnt/0/gru_mfcc_representations.npy', 5168, 'mnt/0/gru_mfcc_distances')

get_PCA_Tf_idf_representations('/mnt/0/pca_tf_idf_model_90_ratio', '/mnt/0/tf_idf_representations.npy', '/mnt/0/pca_tf_idf_representations')