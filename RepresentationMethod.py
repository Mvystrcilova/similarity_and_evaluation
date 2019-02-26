import abc
from Song import Song
import librosa
import numpy, pandas
import librosa.display
import matplotlib.pyplot as plt
from matplotlib import interactive
interactive(True)
from sklearn import decomposition, preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from keras import Sequential, optimizers, Model
from keras.layers import LSTM, Bidirectional, GRU, Input
from keras.models import load_model


adam = optimizers.adam(lr=0.0001, clipnorm=1.)



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
        pass

    def train(self, songs):
        lyrics = []
        for s in songs:
            lyrics.append(s.lyrics)

        tfidf_vectorizer = TfidfVectorizer()
        tfidf_train_matrix = (tfidf_vectorizer.fit_transform(songs))

        for i,s in enumerate(songs):
            s.tf_idf_representation = tfidf_train_matrix[i]


class Word2Vec(TextMethod):

    def __init__(self, w2v_model, stopwords=[]):
        self.w2v_model = w2v_model
        self.stopwords = stopwords

    def represent_song(self, song):
        lyrics = song.lyrics.lower()
        words = [w.strip('.,!:?-') for w in lyrics.split(" ") if w.strip(".,!?:-") not in self.stopwords]
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

    def train(self, songs):
        # the model for W2V is already pretrained by Google so no need to implement this
        pass


class AudioMethod(RepresentationMethod):

    def get_spectrogram(self, song):
        filename = song.file_path
        y, sr = librosa.load(filename, offset=15, duration=60)

        return y, sr


class MFCCRepresentation(AudioMethod):

    def represent_song(self, song):
        MFCC = librosa.feature.mfcc(y=self.get_spectrogram(song).y, sr=self.get_spectrogram(song).sr, n_mels=128, fmax=8000).flatten()
        return MFCC

    def train(self, songs):
        song_frame = pandas.DataFrame()
        for s in songs:
            mfcc = self.represent_song(s)
            temp_df = pandas.DataFrame(data=mfcc)
            song_frame.append(temp_df)

        for i, s in enumerate(songs):
            s.mfcc_representation = song_frame[i]


class PCA_Mel_spectrogram(AudioMethod):

    def represent_song(self, song):
        mel_spectrogram = librosa.feature.melspectrogram(y=self.get_spectrogram(song).y, sr=self.get_spectrogram(song).sr, n_mels=128, fmax=8000).flatten()
        return mel_spectrogram.flatten()

    def train(self, songs):
        song_frame = pandas.DataFrame()
        for s in songs:
            spec = self.represent_song(s)
            temp_df = pandas.DataFrame(data=spec)
            song_frame.append(temp_df)

        pca = decomposition.PCA()
        song_frame_afterPCA = pca.fit_transform(song_frame)
        for i, s in enumerate(songs):
            s.mel_pca_representation = song_frame_afterPCA[i]


class PCA_Spectrogram(AudioMethod):

    def extract_audio(self, song):
        spec = numpy.abs(librosa.stft(self.get_spectrogram(song).y)).flatten()
        return spec


    def train(self, songs):
        song_frame = pandas.DataFrame()
        for s in songs:
            spec = self.represent_song(s)
            temp_df = pandas.DataFrame(data=spec)
            song_frame.append(temp_df)

        pca = decomposition.PCA()
        song_frame_afterPCA = pca.fit_transform(song_frame)
        for i,s in enumerate(songs):
            s.pca_representation = song_frame_afterPCA[i]


class LSTM_Mel_Spectrogram(AudioMethod):

    def __init__(self):
        self.model_name = 'LSTM_Mel_model.h5'
        self.time_stamps = 136
        self.features = 320

    def extract_audio(self, song):
        y, sr = librosa.load(song.file_path)
        mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=320, n_fft=4410, hop_length=812)
        return mel_spectrogram

    def normalize_input(self, mel_spectrogram):
        normalized_input = normalize(mel_spectrogram, axis=1).reshape(len(mel_spectrogram[0]), len(mel_spectrogram))
        return normalized_input

    def train(self, songs):
        model = Sequential()
        model.add(LSTM(160, activation='sigmoid', return_sequences=True, input_shape=(136, 320)))
        model.add(LSTM(80, activation='sigmoid', return_sequences=True, input_shape=(136, 160)))
        model.add(Bidirectional(LSTM(160, activation='tanh', return_sequences=True)))
        model.compile(optimizer=adam, loss='mse')

        model.summary()
        input_songs = []
        for s in songs:
            input_song = self.normalize_input(self.extract_audio(s))
            input_songs.append(input_song)

        model.fit(numpy.array(input_songs), numpy.array(input_songs), epochs=100)

        encoder = Model(inputs=model.input, outputs=model.get_layer(index=1).output)
        encoder.save(self.model_name)
        # model.fit(normalized_input, normalized_input, epochs=1)

    def get_model(self):
        return load_model(self.model_name)

    def represent_song(self, song):
        return self.get_model.predict(self.extract_audio(song).reshape(1, self.time_stamps, self.features))[0, :, 0]


class GRU_Mel_Spectrogram(AudioMethod):
    def __init__(self):
        self.model_name = 'GRU_Mel_model.h5'
        self.time_stamps = 136
        self.features = 320

    def extract_audio(self, song):
        y, sr = librosa.load(song.file_path)
        mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=320, n_fft=4410, hop_length=812)
        return mel_spectrogram

    def normalize_input(self, mel_spectrogram):
        normalized_input = normalize(mel_spectrogram, axis=1).reshape(self.time_stamps, self.features)
        return normalized_input

    def train(self, songs):
        encoder_inputs = Input(shape=(self.time_stamps, self.features), name='input')
        encoded = GRU(int(self.features/2), return_sequences=True)(encoder_inputs)
        encoded = GRU(int(self.features/4), return_sequences=True)(encoded)
        decoded = Bidirectional(GRU(int(self.features/2),
                                    activation='tanh',
                                    return_sequences=True,
                                    name='output'))(encoded)

        auto_encoder = Model(encoder_inputs, decoded)
        encoder = Model(encoder_inputs, encoded)

        auto_encoder.summary()
        encoder.summary()

        input_songs = []
        for s in songs:
            input_song = self.normalize_input(self.extract_audio(s))
            input_songs.append(input_song)

        auto_encoder.fit(numpy.array(input_songs), numpy.array(input_songs), epochs=100)
        encoder.save(self.model_name)

    def get_model(self):
        return load_model(self.model_name)

    def represent_song(self, song):
        return self.get_model.predict(
                self.extract_audio(song).reshape(
                1, self.time_stamps, self.features))[0, :, 0]


class GRU_Spectrogram(AudioMethod):
    def __init__(self):
        self.model_name = 'GRU_Spec_model.h5'
        self.time_stamps = 136
        self.features = 2206

    def extract_audio(self, song):
        y, sr = librosa.load(song.file_path)
        spectrogram = librosa.core.stft(y=y, n_fft=4410, hop_length=812)
        return spectrogram

    def normalize_input(self, spectrogram):
        normalized_input = normalize(spectrogram, axis=1).reshape(self.time_stamps, self.features)
        return normalized_input

    def train(self, songs):
        encoder_inputs = Input(shape=(self.time_stamps, self.features), name='input')
        encoded = GRU(int(self.features/4), return_sequences=True)(encoder_inputs)
        encoded = GRU(int(self.features/7), return_sequences=True)(encoded)
        decoded = Bidirectional(GRU(int(self.features/2),
                                    activation='softmax',
                                    return_sequences=True,
                                    name='output'))(encoded)

        auto_encoder = Model(encoder_inputs, decoded)
        encoder = Model(encoder_inputs, encoded)

        auto_encoder.summary()
        encoder.summary()

        input_songs = []
        for s in songs:
            input_song = self.normalize_input(self.extract_audio(s))
            input_songs.append(input_song)

        auto_encoder.fit(numpy.array(input_songs), numpy.array(input_songs), epochs=100)
        encoder.save(self.model_name)

    def get_model(self):
        return load_model(self.model_name)

    def represent_song(self, song):
        return self.get_model.predict(
                self.extract_audio(song).reshape(
                1, self.time_stamps, self.features))[0, :, 0]

    class LSTM_Spectrogram(AudioMethod):

        def __init__(self):
            self.model_name = 'LSTM_Spec_model.h5'
            self.time_stamps = 136
            self.features = 320

        def extract_audio(self, song):
            y, sr = librosa.load(song.file_path)
            mel_spectrogram = librosa.core.stft(y=y, n_fft=4410, hop_length=812)
            return mel_spectrogram

        def normalize_input(self, spectrogram):
            normalized_input = normalize(spectrogram, axis=1).reshape(self.time_stamps, self.features)
            return normalized_input

        def train(self, songs):
            model = Sequential()
            model.add(LSTM(160, activation='sigmoid', return_sequences=True, input_shape=(self.time_stamps, self.features)))
            model.add(LSTM(80, activation='sigmoid', return_sequences=True, input_shape=(self.time_stamps, self.features)))
            model.add(Bidirectional(LSTM(160, activation='tanh', return_sequences=True)))
            model.compile(optimizer=adam, loss='mse')

            model.summary()
            input_songs = []
            for s in songs:
                input_song = self.normalize_input(self.extract_audio(s))
                input_songs.append(input_song)

            model.fit(numpy.array(input_songs), numpy.array(input_songs), epochs=100)

            encoder = Model(inputs=model.input, outputs=model.get_layer(index=1).output)
            encoder.save(self.model_name)
            # model.fit(normalized_input, normalized_input, epochs=1)

        def get_model(self):
            return load_model(self.model_name)

        def represent_song(self, song):
            return self.get_model.predict(self.extract_audio(song).reshape(1, self.time_stamps, self.features))[0, :, 0]




