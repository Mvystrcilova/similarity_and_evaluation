import numpy, pandas, librosa, os, sklearn.preprocessing

def file_to_spectrogram(filename, n_fft, hop_length):
    y,sr = librosa.load(filename)
    spectrogram = numpy.abs(librosa.core.stft(y, n_fft=n_fft, hop_length=hop_length))
    return spectrogram

def convert_files_to_specs(directory, n_fft, hop_length):
    all_songs = pandas.read_csv(directory, header=None, sep=';', index_col=False, names=['artist', 'title', 'lyrics', 'link', 'path'])
    scaler = sklearn.preprocessing.MinMaxScaler()
    specs = numpy.empty(shape=[16594,2206,408])
    for i, song in all_songs.iterrows():
        filename = song['path'].split('/')
        wav_file = '/Users/m_vys/PycharmProjects/cleaned_wav_files/' + filename[5][:-3] + 'wav'
        spectrogram = file_to_spectrogram(wav_file, n_fft, hop_length)
        spectrogram = scaler.fit_transform(spectrogram)
        specs[i] = spectrogram
        print(i)

    numpy.save('song_spectrograms', specs)

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

    numpy.save('song_mel_spectrograms', mels)

def convert_spectrograms_to_mfcc(spectrogram_representations, ):
    vectors = numpy.load(spectrogram_representations)
    mfcc_representations = numpy.empty([16594, 0,0])
    scaler = sklearn.preprocessing.MinMaxScaler
    for i in range(16594):
        mfcc_representation = librosa.feature.mfcc(vectors[i], sr=22050, n_mels = 320)
        mfcc_representation = scaler.fit_transform(mfcc_representation)
        mfcc_representations[i] = mfcc_representation

    numpy.save('mfcc_representations', mfcc_representations)

