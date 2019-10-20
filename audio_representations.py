import numpy, pandas, librosa, os, sklearn.preprocessing, joblib, glob, pickle
# from keras.models import model_from_json, load_model
import re
import urllib.request
from bs4 import BeautifulSoup
import youtube_dl
import shutil

from pydub import AudioSegment
from things_i_hopefully_wont_use_anymore.convert_to_wav import create_song_segment
# def numericalSort(value):
#     parts = numbers.split(value)
#     parts[1::2] = map(int, parts[1::2])
#     return parts


def file_to_spectrogram(filename, n_fft, hop_length):
    y,sr = librosa.load(filename)
    spectrogram = numpy.abs(librosa.core.stft(y, n_fft=n_fft, hop_length=hop_length))
    return spectrogram


def convert_files_to_specs(directory, n_fft, hop_length):
    all_songs = pandas.read_csv(directory, header=None, sep=';', index_col=False, names=['artist', 'title', 'lyrics',
                                                                                         'link', 'path'])
    scaler = sklearn.preprocessing.MinMaxScaler()
    # specs = numpy.empty(shape=[16594,2206,408])
    for i, song in all_songs.iterrows():
        if i >= 12247:
            filename = song['path'].split('/')
            wav_file = '/Users/m_vys/PycharmProjects/cleaned_wav_files/' + filename[5][:-3] + 'wav'
            spectrogram = file_to_spectrogram(wav_file, n_fft, hop_length)
            spectrogram = scaler.fit_transform(spectrogram)
            numpy.save('/Users/m_vys/PycharmProjects/similarity_and_evaluation/spectrograms/' + str(i) + '_' +
                       str(song['artist'] + ' - ' + str(song['title'])), spectrogram)
            print(i)


def file_to_mel(filename, nmels, n_fft, hop_length):
    y,sr = librosa.load(filename)
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=320, n_fft=4410, hop_length=812)
    return mel_spectrogram

def download_from_youtube(row):

    textToSearch = row['artist'] + ' ' + row['title']
    query = urllib.parse.quote(textToSearch)
    url = "https://www.youtube.com/results?search_query=" + query
    response = urllib.request.urlopen(url)
    html = response.read()
    soup = BeautifulSoup(html, 'html.parser')
    # for vid in soup.findAll(attrs={'class':'yt-uix-tile-link'}):
    #     print('https://www.youtube.com' + vid['href'])
    vid = soup.findAll(attrs={'class':'yt-uix-tile-link'})[0]
    l = 'https://www.youtube.com' + vid['href']

    name = str(row['path'].split('/')[1][:-4])
    # regex = re.compile('[^A-Za-z0-9. -]')
    # name = regex.sub("", name)
    print(name)

    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],

        'outtmpl': '/Volumes/LaCie/missing_mp3_files/' + name + '.%(ext)s'
    }
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        ydl.download([l])
        info_dict = ydl.extract_info(l, download=False)
        print(info_dict.get('filename', None))
        print(row['path'])
    # df.at[i,'link_on_disc']

def convert_files_to_mels_and_mfccs(directory, n_mels, n_fft, hop_length, n_mfcc):
    all_songs = pandas.read_csv(directory, header=None, sep=';', index_col=False, names=['artist', 'title', 'lyrics',
                                                                          'link', 'path'], engine='python')
    song_paths = all_songs['path']
    # song_paths = song_paths.sort_values()
    # song_paths.to_csv('song_paths', header=None, index=False, sep=';')
    scaler = sklearn.preprocessing.MinMaxScaler()
    mels = numpy.empty(shape=[16594, 320, 815])
    mfccs = numpy.empty(shape=[16594, 128, 1292])
    j = 0
    missing_songs = open('missing_songs.txt', "a+" )
    number_of_hopelessly_missing_songs = 0
    for i, song in all_songs.iterrows():

        # if i >= 6047:
        #     print(i)
            filename_array = song['path'].split('/')
            # wav_file = '/Users/m_vys/PycharmProjects/cleaned_wav_files/' + filename[5][:-3] + 'wav'
            song_file = '/Volumes/LaCie/used_mp3_files/' + filename_array[1]
            moved_song_file = '/Volumes/LaCie/used_mp3_files/' + filename_array[1]
            if os.path.isfile(song_file) or os.path.isfile(moved_song_file):
                print(i)
                if not os.path.isfile(moved_song_file):
                    shutil.move(song_file, '/Volumes/LaCie/used_mp3_files/' + filename_array[1])

                try:
                    print(filename_array[1])
                    wav_file = create_song_segment(song_file,i, missing_songs)
                    wav_song = librosa.load(wav_file)
                    mel_spectrogram = file_to_mel(wav_file, n_mels, n_fft, hop_length)
                    mfcc_representation = librosa.feature.mfcc(wav_song[0], wav_song[1], n_mfcc=n_mfcc)
                    mel_spectrogram = scaler.fit_transform(mel_spectrogram)
                    mfcc_representation = scaler.fit_transform(mfcc_representation)
                    mels[i] = mel_spectrogram
                    mfccs[i] = mfcc_representation
                    os.remove(wav_file)
                    print(i)
                except Exception as e:
                    mels[i] = numpy.zeros([320, 815])
                    mfccs[i] = numpy.zeros([128, 1292])
                    message = 'unable to turn to specs ' + str(j) + 'th songfile ' + song_file + ' on index ' + str(i) + 'substituted with zero array' + '\n'
                    missing_songs.write(message)
                    j += 1
                    print(message)

            else:
                try:
                    download_from_youtube(song)
                except Exception as e:
                    print(e)
            #         print('download of song failed')
            #         print(song)
            #         number_of_hopelessly_missing_songs += 1
                mels[i] = numpy.zeros([320, 815])
                mfccs[i] = numpy.zeros([128, 1292])
                message = 'missing ' + str(j) +'th songfile ' + song_file + ' on index ' + str(i) + '\n'
                missing_songs.write(message)
                j += 1
                print(message)


    # print(str(number_of_hopelessly_missing_songs))
    numpy.save('mel_spectrograms_30sec', mels)
    numpy.save('mfccs_30sec', mfccs)
    missing_songs.close()




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


# def get_PCA_Spec_representations(pca_model, directory, repr_name):
#     representations = numpy.empty([16594, 320])
#     pca_model = joblib.load(pca_model)
#     i = 0
#     for file in sorted(glob.glob(directory + '/*.npy'), key=numericalSort):
#         spec = numpy.load(file).reshape([1,900048])
#         pca_spec = pca_model.transform(numpy.abs(spec))
#         print(i, pca_spec.shape)
#         representations[i] = pca_spec
#         i = i+1
#     numpy.save(repr_name, representations)

# def get_nn_Spec_representations(nn_model, directory, repr_name):
#     representations = numpy.empty([16594, 5712])
#     model = load_model(nn_model)
#     i = 0
#     for file in sorted(glob.glob(directory + '/*.npy'), key=numericalSort):
#         spec = numpy.load(file).reshape([1, 408, 2206])
#         nn_spec = model.predict(numpy.abs(spec))[0]
#         print(i, nn_spec.shape)
#         representations[i] = nn_spec.reshape([1, 5712])
#         i = i + 1
#
#     numpy.save(repr_name, representations)


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

# def find_missing_songs(directory):
#     all_songs = pandas.read_csv(directory, header=None, sep=';', index_col=False, names=['artist', 'title', 'lyrics',
#                                                                                          'link', 'path'],
#                                 engine='python')
#     mels = numpy.load('mel_spectrograms_30sec')
#     mfccs = numpy.load('mfccs_30sec')
#     for i, song in all_songs.iterrows():
#         if mels[i] == numpy.zeros()
#
# def get_tf_idf_representations(tf_idf_model, songs, tf_idf_numpy_filename):
#     vectorizer = pickle.load(open(tf_idf_model, 'rb'))

# get_PCA_Spec_representations('/mnt/0/big_pca_model', 'mnt/0/spectrograms', 'mnt/0/pca_spec_representations_1106')
# get_PCA_Mel_representations('mnt/0/mel_spec_pca_model_90_ratio', 'mnt/0/song_mel_spectrograms.npy',
# 'mnt/0//mel_spec_representations_5717')
#convert_files_to_mels('not_empty_songs', 329, 4410, 812)
# convert_files_to_mfcc('not_empty_songs', 320)

# get_MFCC_representations('/mnt/0/models/GRU_MFCC_model.json', '/mnt/0/GRU_MFCC_model.h5', '/mnt/0/mfcc_representations.npy', '/mnt/0/gru_mfcc_representations')
# print('gru_mfcc representations saved')
# get_MFCC_representations('/mnt/0/LSTM_MFCC_model.json', '/mnt/0/LSTM_MFCC_model.h5', '/mnt/0/mfcc_representations.npy', '/mnt/0/lstm_mfcc_representations')
# print('lstm mfcc representations saved')

# from Distances import save_neural_network
#
# save_neural_network('mnt/0/gru_mfcc_representations.npy', 5168, 'mnt/0/gru_mfcc_distances')

convert_files_to_mels_and_mfccs('/Users/m_vys/PycharmProjects/similarity_and_evaluation/not_empty_songs_relative_path.txt', 320, 4410, 812, 320)