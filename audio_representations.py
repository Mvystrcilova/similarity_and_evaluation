import numpy, pandas, librosa, os, sklearn.preprocessing, joblib, glob, pickle

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

    pca_mel_representations = numpy.empty([16594, 5715])
    for i in range(16594):
        mel_spec = mel_specs[i].reshape([1, 130560])
        mel_spec = scaler.fit_transform(mel_spec)
        pca_mel_spec = model.transform(mel_spec)
        pca_mel_representations[i] = pca_mel_spec
        print(i)

    numpy.save(repr_name, pca_mel_representations)


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

def get_tf_idf_representations(tf_idf_model, songs, tf_idf_numpy_filename):
    vectorizer = pickle.load(open(tf_idf_model, 'rb'))

get_PCA_Spec_representations('/mnt/0/big_pca_model', 'mnt/0/spectrograms', 'mnt/0/pca_spec_representations_1106')
# get_PCA_Mel_representations('models/mel_spec_pca_model_90_ratio', 'representations/song_mel_spectrograms.npy', 'representations/mel_spec_representations_5717')
#convert_files_to_mels('not_empty_songs', 329, 4410, 812)
# convert_files_to_mfcc('not_empty_songs', 320)