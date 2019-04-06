import abc
# from Song import Song
# import librosa
import numpy, pandas, scipy, sklearn
import librosa.display
from matplotlib import interactive
interactive(True)
from sklearn import decomposition, preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from keras import Sequential, optimizers, Model
from keras.layers import LSTM, Bidirectional, GRU, Input
from keras.models import load_model

import math
import pickle, glob, joblib
from things_i_hopefully_wont_use_anymore.minisom import MiniSom
# from Evaluation import Evaluation
# from Dataset import Dataset
from gensim.models.keyedvectors import KeyedVectors

import re
numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts


adam = optimizers.adam(lr=0.0001, clipnorm=1.)
w2v_model = 'model to be inserted'


class RepresentationMethod(abc.ABC):
    @abc.abstractmethod
    def represent_song(self, song):
        pass

    @abc.abstractmethod
    def train(self, songs):
        pass


class TextMethod(RepresentationMethod):
    pass


class TF_idf(TextMethod):

    def represent_song(self, song):
        # the songs need to be represented all together in order to get all the words so there is no
        # reason to put a bag of words representation here
        # if one song is added, the whole matrix will be updated
        pass

    def train(self, songs):
        lyrics = []
        for s in songs:
            lyrics.append(s.lyrics)

        tfidf_vectorizer = TfidfVectorizer()
        tfidf_train_matrix = (tfidf_vectorizer.fit_transform(lyrics))
        tf_idf_representations = numpy.empty([16594, tfidf_train_matrix[0].toarray().size])
        for i, s in enumerate(songs):
            l = [lyrics[i]]
            prediction = tfidf_vectorizer.transform(l)
            print(prediction)
            tf_idf_representations[i] = prediction.toarray()

        scipy.sparse.save_npz('tf_idf_distance_matrix', tfidf_train_matrix)
        pickle.dump(tfidf_vectorizer, open('tfidf_model', 'wb'))
        numpy.save('tf_idf_representations', tf_idf_representations)


class Word2Vec(TextMethod):

    def __init__(self, stopwords=[]):
        self.stopwords = stopwords
        self.w2v_model = KeyedVectors.load('/Users/m_vys/Documents/matfyz/rocnikac/djangoApp/rocnikac/w2v_subset', mmap='r')

    def represent_song(self, song):
        lyrics = song.lyrics.lower()
        words = [w.strip('.,!:?-') for w in lyrics.split(' ') if w.strip('.,!?:-') not in self.stopwords]
        word_vecs = []
        for word in words:
            try:
                vec = self.w2v_model[word]
                word_vecs.append(vec)

            except KeyError:
                # Ignore, if the word doesn't exist in the vocabulary
                pass

        # Assuming that document vector is the mean of all the word vectors
        # PS: There are other & better ways to do it.
        vector = numpy.mean(word_vecs, axis=0)
        song.W2V_representation = vector
        return vector

    def train(self, songs):
        # the model for W2V is already pretrained by Google so no need to implement this
        pass


class SOM_TF_idf(TextMethod):
    def __init__(self, sigma, learning_rate):
        self.sigma = sigma
        self.learning_rate = learning_rate
        self.model_name = 'som_tf_idf.p'


    def train(self, songs):
        train_data = pandas.DataFrame()
        # the representation is used from the TF-idf method
        for s in songs:
            song_representation = pandas.DataFrame(data=[[s.tf_idf_representation]])
            train_data = train_data.append(song_representation)

        scaler = preprocessing.MinMaxScaler()
        train_data = scaler.fit_transform(train_data)
        grid_size = int(5*(math.sqrt(len(songs))))
        som = MiniSom(grid_size, grid_size, len(songs[0].tf_idf_representation))
        som.random_weights_init(train_data)
        som.train_random(train_data,num_iteration=len(songs)*10)

        for s in songs:
            s.som_tf_idf_representation = som.winner(s.tf_idf_representation)
        with open(self.model_name, 'wb') as outfile:
            pickle.dump(som, outfile)

    def represent_song(self, song):
        # tf_idf_repr = Word2Vec(w2v_model, stopwords=[])
        # tf_idf_repr.represent_song(song)
        # with open(self.model_name, 'rb') as infile:
        #     som = pickle.load(infile)
        # song.som_w2v_representation = som.winner(song.W2V_representation)
        pass


class SOM_W2V(TextMethod):
    def __init__(self, sigma, learning_rate, grid_size_multiple, iterations, model_name):
        self.sigma = sigma
        self.learning_rate = learning_rate
        self.model_name = model_name
        self.grid_size_multiple = grid_size_multiple,
        self.iterations = iterations
        self.w2v_model = Word2Vec([])

    def train(self, songs):
        train_data = []
        # the representation is used from the W2V method
        for s in songs:
            song_representation = self.w2v_model.represent_song(s)
            train_data.append(song_representation)

        scaler = preprocessing.MinMaxScaler()
        lenght = len(train_data)
        train_data = scaler.fit_transform(train_data)
        # train_data = pandas.DataFrame(train_data)
        grid_size = int(self.grid_size_multiple[0]) * int(math.sqrt(lenght))
        som = MiniSom(grid_size, grid_size, som_model_name=self.model_name, input_len=300, )
        som.random_weights_init(train_data)
        som.train_batch(train_data, num_iteration=(lenght * self.iterations), verbose=True)
        with open(self.model_name, 'wb') as outfile:
            pickle.dump(som, outfile)
        for s in songs:
            s.som_w2v_representation = som.winner(s.W2V_representation)

    # song representations are defined during training
    def represent_song(self, song):
        w2v_repr = Word2Vec(self.w2v_model, stopwords=[])
        w2v_repr.represent_song(song)
        with open(self.model_name, 'rb') as infile:
            som = pickle.load(infile)
        song.som_w2v_representation = som.winner(song.W2V_representation)
        pass

    def load_representation(self, representation_place):
        songs_from_file = pandas.read_csv(representation_place, sep=';',
                                names=['somethingWeird',
                                       'songId', 'title',
                                       'artist', 'tf_idf_representation',
                                       'W2V_representation'],
                                usecols=['songId', 'title', 'artist',
                                         'W2V_representation'], header=None)
        for i, s in songs_from_file.iterrows():
            s.W2V_representation = s['W2V_representation']

        return songs_from_file


class AudioMethod(RepresentationMethod):

    def get_spectrogram(self, song):
        filename = song.file_path
        y, sr = librosa.load(filename, offset=15, duration=60)

        return y, sr


class MFCCRepresentation(AudioMethod):

    def represent_song(self, song):
        MFCC = librosa.feature.mfcc(y=self.get_spectrogram(song).y, sr=self.get_spectrogram(song).sr, n_mels=320, fmax=8000).flatten()
        return MFCC

    def train(self, songs):
        song_frame = pandas.DataFrame()
        for s in songs:
            mfcc = self.represent_song(s)
            temp_df = pandas.DataFrame(data=mfcc)
            song_frame = song_frame.append(temp_df)

        for i, s in enumerate(songs):
            s.mfcc_representation = song_frame[i]


class PCA_Mel_spectrogram(AudioMethod):

    def __init__(self, mel_spectrograms):
        self.mel_spectrograms = mel_spectrograms

    def represent_song(self, song):
        mel_spectrogram = librosa.feature.melspectrogram(y=self.get_spectrogram(song).y, sr=self.get_spectrogram(song).sr, n_mels=128, fmax=8000).flatten()
        return mel_spectrogram.flatten()

    def train(self, songs):
        mel_spectrograms = numpy.load('mnt/0/song_mel_spectrograms.npy').reshape([16594, 130560])
        ipca = decomposition.IncrementalPCA(n_components=3000, batch_size=30)
        for j in range(1,int(16594/3000)):
                ipca.partial_fit(mel_spectrograms[int((j-1)*3000):int(j*3000)])
                print(j, 'chunk fitted')
        try:
            joblib.dump(ipca, 'mnt/0/big_mel_pca_model')
        except:
            file_path = 'spec_pca_model.pkl'
            max_bytes = 2 ** 31 - 1

            ## write
            bytes_out = pickle.dumps(ipca)
            with open(file_path, 'wb') as f_out:
                for idx in range(0, len(bytes_out), max_bytes):
                    f_out.write(bytes_out[idx:idx + max_bytes])

        # for i, s in enumerate(songs):
        #     s.mel_pca_representation = song_frame_afterPCA[i]
    def train_normal_PCA(self):
        arrays = numpy.load('/mnt/0/song_mel_spectrograms.npy').reshape([16594, 130560])
        print('mels loaded')
        pca = decomposition.PCA(n_components=5715)
        print('pca created')
        pca.fit(arrays)
        joblib.dump(pca, '/mnt/0/mel_spec_pca_model_90_ratio')


class PCA_Spectrogram(AudioMethod):

    def __init__(self, spec_directory):
        self.spec_directory = spec_directory

    def extract_audio(self, song):
        spec = numpy.abs(librosa.stft(self.get_spectrogram(song).y)).flatten()
        return spec

    def represent_song(self, song):
        pass

    def train_with_a_lot_of_memory(self):
        arrays = numpy.empty([16594, 900048])
        ipca = decomposition.IncrementalPCA(n_components=1106, batch_size=20)
        i = 0
        for file in sorted(glob.glob(self.spec_directory + '/*.npy'), key=numericalSort):
            array = numpy.load(file)
            array = array.reshape([1, 900048])
            print(i)
            arrays[i] = array
            i = i + 1
        for j in range(1, int(16594 / 1106)):
            ipca.partial_fit(arrays[int((j - 1) * 1106):int(j * 1106)])
            print(j, 'chunk fitted out of', int(16594/1106))
        joblib.dump(ipca, 'mnt/0/pca_spec_model_1106')

    def train(self):
        i = 0
        chunk = numpy.empty([1106, 900048])
        ipca = decomposition.IncrementalPCA(n_components=1106, batch_size=20)
        for file in sorted(glob.glob(self.spec_directory + '/*.npy'), key=numericalSort):
            if (i % 1106 != 0) or (i == 0):
                array = numpy.load(file)
                array = array.reshape([1, 900048])
                print(i, str(i % 1106))
                chunk[i % 1106] = array
            else:
                ipca.partial_fit(chunk)
                print('chunk fitted')
            i = i+1
        try:
            joblib.dump(ipca, '/mnt/0/big_spec_pca_model')
        except:
            file_path = 'spec_pca_model.pkl'
            max_bytes = 2 ** 31 - 1

            ## write
            bytes_out = pickle.dumps(ipca)
            with open(file_path, 'wb') as f_out:
                for idx in range(0, len(bytes_out), max_bytes):
                    f_out.write(bytes_out[idx:idx + max_bytes])

    def train_normal_PCA(self):
        i = 0
        arrays = numpy.empty([16594, 900048])

        for file in sorted(glob.glob(self.spec_directory + '/*.npy'), key=numericalSort):
            array = numpy.load(file)
            array = array.reshape([1, 900048])
            print(i)
            arrays[i] = array
            i = i + 1

        pca = decomposition.PCA()
        pca.fit(arrays)
        joblib.dump(pca, '/mnt/0/normal_spec_pca_model')


class LSTM_Mel_Spectrogram(AudioMethod):

    def __init__(self):
        self.model_name = '/mnt/0/models/final_LSTM_Mel_model.h5'
        self.time_stamps = 408
        self.features = 320

    def extract_audio(self, song):
        y, sr = librosa.load(song.file_path)
        mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=320, n_fft=4410, hop_length=812)
        return mel_spectrogram

    def normalize_input(self, mel_spectrogram):
        sc = MinMaxScaler((-1,1))
        normalized_input = sc.fit_transform(mel_spectrogram).reshape(len(mel_spectrogram[0]), len(mel_spectrogram))
        return normalized_input

    def train(self, songs):
        model = Sequential()
        model.add(LSTM(int(self.features/11), activation='sigmoid', return_sequences=True, input_shape=(self.time_stamps, self.features)))
        model.add(LSTM(int(self.features/22), activation='sigmoid', return_sequences=True, input_shape=(self.time_stamps, self.features)))
        model.add(Bidirectional(LSTM(160, activation='tanh', return_sequences=True)))
        model.compile(optimizer=adam, loss='mse')

        model.summary()

        input_songs = numpy.load('/mnt/0/song_mel_spectrograms.npy').reshape([16594, 408, 320])

        # train_X, train_y, test_X, test_y = sklearn.model_selection.train_test_split(input_songs, input_songs,
        #                                                                             test_size=0.2, random_state=13)
        model.compile(adam, loss='mse')
        model.fit(input_songs, input_songs, batch_size=256, epochs=150)
        encoder = Model(inputs=model.input, outputs=model.get_layer(index=1).output)

        encoder.compile(adam, loss='mse')
        encoder.save(self.model_name)
        model.save('/mnt/0/models/lstm_mel_spec_autoencoder2.h5')
        model_json = encoder.to_json()
        with open("/mnt/0/LSTM_Mel_model.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        encoder.save_weights("/mnt/0/LSTM_Mel_model.h5")
        print("Saved LSTM Mel model to disk")

    def get_model(self):
        return load_model(self.model_name)

    def represent_song(self, song):
        return self.get_model.predict(self.extract_audio(song).reshape(1, self.time_stamps, self.features))[0, :, 0]


class GRU_Mel_Spectrogram(AudioMethod):
    def __init__(self):
        self.model_name = 'mnt/0/models/final_GRU_Mel_model.h5'
        self.time_stamps = 408
        self.features = 320


    def extract_audio(self, song):
        y, sr = librosa.load(song.file_path)
        mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=320, n_fft=4410, hop_length=812)
        return mel_spectrogram

    def normalize_input(self, mel_spectrogram):
        sc = MinMaxScaler((-1, 1))
        normalized_input = sc.fit_transform(mel_spectrogram).reshape(self.time_stamps, self.features)
        return normalized_input

    def train(self, songs):
        encoder_inputs = Input(shape=(self.time_stamps, self.features), name='input')
        encoded = GRU(int(self.features/11), return_sequences=True)(encoder_inputs)
        encoded = GRU(int(self.features/22), return_sequences=True)(encoded)
        decoded = Bidirectional(GRU(int(self.features/2),
                                    activation='tanh',
                                    return_sequences=True,
                                    name='output'))(encoded)

        auto_encoder = Model(encoder_inputs, decoded)
        encoder = Model(encoder_inputs, encoded)

        auto_encoder.summary()
        encoder.summary()

        input_songs = numpy.load('/mnt/0/song_mel_spectrograms.npy').reshape([16594,408,320])

        auto_encoder.compile(adam, loss='mse')
        auto_encoder.fit(input_songs, input_songs, batch_size=256, epochs=150)
        encoder.save(self.model_name)
        auto_encoder.save('/mnt/0/models/gru_spec_autoencoder.h5')
        model_json = encoder.to_json()
        with open("/mnt/0/GRU_Mel_model.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        encoder.save_weights("/mnt/0/GRU_Mel_model.h5")
        print("Saved GRU Mel model to disk")

    def get_model(self):
        return load_model(self.model_name)

    def represent_song(self, song):
        return self.get_model.predict(
                self.extract_audio(song).reshape(
                1, self.time_stamps, self.features))[0, :, 0]


class GRU_Spectrogram(AudioMethod):
    def __init__(self, spec_directory):
        self.model_name = '/mnt/0/models/final_GRU_Spec_model.h5'
        self.time_stamps = 408
        self.features = 2206
        self.spec_directory = spec_directory

    def extract_audio(self, song):
        y, sr = librosa.load(song.file_path)
        spectrogram = librosa.core.stft(y=y, n_fft=4410, hop_length=812)
        return spectrogram

    def normalize_input(self, spectrogram):
        sc = MinMaxScaler((-1, 1))
        normalized_input = sc.fit_transform(spectrogram).reshape(self.time_stamps, self.features)
        return normalized_input

    def train(self, songs):
        encoder_inputs = Input(shape=(self.time_stamps, self.features), name='input')
        encoded = GRU(int(self.features/22), return_sequences=True)(encoder_inputs)
        encoded = GRU(int(self.features/44), return_sequences=True)(encoded)
        decoded = Bidirectional(GRU(int(self.features/2),
                                    activation='softmax',
                                    return_sequences=True,
                                    name='output'))(encoded)

        auto_encoder = Model(encoder_inputs, decoded)
        encoder = Model(encoder_inputs, encoded)

        auto_encoder.summary()
        encoder.summary()

        auto_encoder.compile(adam, loss='mse')
        encoder.compile(adam, loss='mse')
        trainGen = generate_spectrograms(spec_directory=self.spec_directory, batch_size=295, mode="train")
        auto_encoder.fit_generator(trainGen, steps_per_epoch=56, epochs=50)


        # tbCallBack = keras.callbacks.TensorBoard(log_dir='~/evaluation_project/similarity_and_evaluation/Graph', histogram_freq=0,
        #                             write_graph=True, write_images=True)

        auto_encoder.save('/mnt/0/models/gru_spec_autoencoder.h5')
        encoder.save(self.model_name)
        model_json = encoder.to_json()
        with open("/mnt/0/GRU_Spec_model.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        encoder.save_weights("/mnt/0/GRU_Spec_model.h5")
        print("Saved GRU Spec model to disk")

    def get_model(self):
        return load_model(self.model_name)

    def represent_song(self, song):
        return self.get_model.predict(
                self.extract_audio(song).reshape(
                1, self.time_stamps, self.features))[0, :, 0]


class LSTM_Spectrogram(AudioMethod):

    def __init__(self, spec_directory):
        self.model_name = '/mnt/0/models/final_LSTM_Spec_model.h5'
        self.time_stamps = 408
        self.features = 2206
        self.spec_directory = spec_directory

    def extract_audio(self, song):
        y, sr = librosa.load(song.file_path)
        mel_spectrogram = librosa.core.stft(y=y, n_fft=4410, hop_length=812)
        return mel_spectrogram

    def normalize_input(self, spectrogram):
        sc = MinMaxScaler((-1, 1))
        normalized_input = sc.fit_transform(spectrogram).reshape(self.time_stamps, self.features)
        return normalized_input

    def train(self, songs):
        model = Sequential()
        model.add(LSTM(int(self.features/22), activation='sigmoid', return_sequences=True, input_shape=(self.time_stamps, self.features)))
        model.add(LSTM(int(self.features/44), activation='sigmoid', return_sequences=True, input_shape=(self.time_stamps, self.features)))
        model.add(Bidirectional(LSTM(int(self.features/2), activation='tanh', return_sequences=True)))
        model.compile(optimizer=adam, loss='mse')

        model.summary()

        trainGen = generate_spectrograms(spec_directory=self.spec_directory, batch_size=295, mode="train")
        model.fit_generator(trainGen, steps_per_epoch=56, epochs=50)
        encoder = Model(inputs=model.input, outputs=model.get_layer(index=1).output)
        encoder.save(self.model_name)
        model.save('/mnt/0/models/LSTM_Spec_autoencoder.h5')
        # model.fit(normalized_input, normalized_input, epochs=1)

    def get_model(self):
        return load_model(self.model_name)

    def represent_song(self, song):
        return self.get_model.predict(self.extract_audio(song).reshape(1, self.time_stamps, self.features))[0, :, 0]


def generate_spectrograms(spec_directory, batch_size, mode='train'):
    while True:
        specs = []
        i = 0
        scaler = sklearn.preprocessing.MinMaxScaler()
        files = sorted(glob.glob(spec_directory + '/*.npy'), key=numericalSort)

        while len(specs) < batch_size:
            if (i > 16593):
                i = 0
            spec = numpy.load(files[i]).T
            spec = scaler.fit_transform(spec)
            specs.append(spec)
            i = i+1
        yield ((numpy.array(specs)), numpy.array(specs))


# d = Dataset('[bla]', 'bla')
# songs = d.load_songs('~/Documents/matfyz/rocnikac/data/songs_with_lyrics')
# som_w2v_1 = SOM_W2V(sigma=0.8, learning_rate=0.2, grid_size_multiple=5, iterations=5, model_name='SOM_W2V_batch_5g5i')
# som_w2v_2 = SOM_W2V(sigma=1, learning_rate=0.5, grid_size_multiple=3, iterations=5, model_name='SOM_W2V_batch_3g5i')
# som_w2v_3 = SOM_W2V(sigma=1, learning_rate=0.5, grid_size_multiple=2, iterations=5, model_name='SOM_W2V_batch_2g5i')
#
# som_w2v_1.train(songs)
# som_w2v_2.train(songs)
# som_w2v_3.train(songs)

# pca_spec = PCA_Spectrogram('/mnt/0/spectrograms')
# pca_spec.train_with_a_lot_of_memory()

# pca_mel_spec = PCA_Mel_spectrogram([])
# pca_mel_spec.train_normal_PCA()


